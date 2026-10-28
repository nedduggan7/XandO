import os
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import tensorflow as tf

from game import Game
from game_model import GameCNN


MODEL_PATH = "actor_critic.weights.h5"

# Online learning settings. These are intentionally conservative because
# updates from a single human game can be noisy.
ONLINE_LEARNING_RATE = 0.0001
ONLINE_GAMMA = 0.99
ONLINE_VALUE_COEF = 0.5
ONLINE_ENTROPY_COEF = 0.01
ONLINE_GRADIENT_CLIP_NORM = 1.0


def encode_board(board, player, win_count):
    size = board.shape[0]
    own = (board == player).astype(np.float32)
    opponent = (board == -player).astype(np.float32)
    rule = np.full((size, size), win_count / size, dtype=np.float32)
    return np.stack([own, opponent, rule], axis=-1)


def find_winning_moves(game, player):
    """Return legal moves that immediately win for player."""
    winning_moves = []

    for move in game.get_legal_moves():
        row, col = divmod(move, game.size)
        game.board[row, col] = player
        wins = game.check_win(row, col, player)
        game.board[row, col] = 0

        if wins:
            winning_moves.append(move)

    return winning_moves


def get_tactical_move(game, player):
    """Take an immediate win, otherwise block a single immediate threat."""
    own_wins = find_winning_moves(game, player)
    if own_wins:
        return own_wins[0]

    opponent_wins = find_winning_moves(game, -player)
    if len(opponent_wins) == 1:
        return opponent_wins[0]

    return None


def select_move(model, game, player):
    """Return move, move source, encoded state, and legal moves before the move."""
    state = encode_board(game.board, player, game.win_count)
    legal_moves = np.asarray(game.get_legal_moves(), dtype=np.int32)

    if len(legal_moves) == 0:
        return None, "No legal moves", state, legal_moves

    tactical_move = get_tactical_move(game, player)
    if tactical_move is not None:
        return int(tactical_move), "Tactical move", state, legal_moves

    logits, _ = model(state[np.newaxis, ...], training=False)
    scores = logits[0].numpy().reshape(-1)
    legal_scores = scores[legal_moves]
    move = int(legal_moves[np.argmax(legal_scores)])

    return move, "Neural network", state, legal_moves


def train_from_game(model, optimizer, trajectory, outcome, total_turns):
    """
    Perform one online actor-critic update from the model's moves in a
    completed game. outcome is +1 for a model win, -1 for a loss, and 0 for draw.

    Only moves selected by the neural network are included. Moves forced by
    the tactical override are excluded from the actor update.
    """
    if not trajectory:
        return None

    states = np.stack([item["state"] for item in trajectory]).astype(np.float32)
    actions = np.asarray([item["action"] for item in trajectory], dtype=np.int32)

    # Discount according to how many full-game turns remained after each move.
    returns_np = np.asarray(
        [
            outcome * (ONLINE_GAMMA ** max(0, total_turns - item["turn_number"] - 1))
            for item in trajectory
        ],
        dtype=np.float32,
    )
    returns = tf.convert_to_tensor(returns_np, dtype=tf.float32)

    batch_size = len(states)
    number_of_squares = states.shape[1] * states.shape[2]

    # Reconstruct legal masks from the saved pre-move positions.
    occupied = (
        states[:, :, :, 0] + states[:, :, :, 1]
    ).reshape(batch_size, -1) > 0.5
    legal_mask = tf.convert_to_tensor(~occupied)

    with tf.GradientTape() as tape:
        logits, values = model(states, training=True)
        logits = tf.reshape(logits, (batch_size, number_of_squares))
        logits = tf.where(
            legal_mask,
            logits,
            tf.cast(-1e9, logits.dtype),
        )

        log_probs = tf.nn.log_softmax(logits, axis=-1)
        probabilities = tf.nn.softmax(logits, axis=-1)
        selected_log_probs = tf.gather(
            log_probs, actions, axis=1, batch_dims=1
        )

        advantages = tf.stop_gradient(returns - values)
        actor_loss = -tf.reduce_mean(selected_log_probs * advantages)
        critic_loss = tf.reduce_mean(tf.square(returns - values))
        entropy = -tf.reduce_mean(
            tf.reduce_sum(probabilities * log_probs, axis=-1)
        )

        total_loss = (
            actor_loss
            + ONLINE_VALUE_COEF * critic_loss
            - ONLINE_ENTROPY_COEF * entropy
        )

    gradients = tape.gradient(total_loss, model.trainable_variables)
    pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, model.trainable_variables)
        if gradient is not None
    ]

    if pairs:
        grads, variables = zip(*pairs)
        grads, _ = tf.clip_by_global_norm(
            grads, ONLINE_GRADIENT_CLIP_NORM
        )
        optimizer.apply_gradients(zip(grads, variables))

    return {
        "actor_loss": float(actor_loss.numpy()),
        "critic_loss": float(critic_loss.numpy()),
        "total_loss": float(total_loss.numpy()),
        "trained_moves": len(trajectory),
    }


class FourInARowGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Four in a Row — Neural Network")
        self.root.minsize(440, 490)

        self.game = None
        self.model = None
        self.optimizer = None
        self.human_player = 1
        self.model_player = -1
        self.cell_buttons = []
        self.game_over = True
        self.model_thinking = False
        self.learning_enabled = False
        self.model_trajectory = []
        self.turn_number = 0

        self.size_var = tk.StringVar(value="5")
        self.win_var = tk.StringVar(value="4")
        self.first_var = tk.StringVar(value="human")

        self._build_controls()
        self._show_message("Choose the board settings, then click Start Game.")

    def _build_controls(self):
        settings = ttk.Frame(self.root, padding=10)
        settings.pack(fill="x")

        ttk.Label(settings, text="Board size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.size_var, width=6).grid(
            row=0, column=1, padx=(5, 14)
        )

        ttk.Label(settings, text="Pieces in a row to win:").grid(
            row=0, column=2, sticky="w"
        )
        ttk.Entry(settings, textvariable=self.win_var, width=6).grid(
            row=0, column=3, padx=(5, 0)
        )

        ttk.Label(settings, text="Who plays first?").grid(
            row=1, column=0, sticky="w", pady=(9, 0)
        )
        first_box = ttk.Combobox(
            settings,
            textvariable=self.first_var,
            values=("human", "model"),
            state="readonly",
            width=10,
        )
        first_box.grid(row=1, column=1, sticky="w", pady=(9, 0))
        ttk.Button(settings, text="Start Game", command=self.start_game).grid(
            row=1, column=2, columnspan=2, sticky="ew", pady=(9, 0)
        )

        self.status_var = tk.StringVar(value="")
        ttk.Label(
            self.root,
            textvariable=self.status_var,
            padding=(10, 4),
            anchor="center",
            wraplength=650,
        ).pack(fill="x")

        self.board_frame = ttk.Frame(self.root, padding=(10, 4, 10, 10))
        self.board_frame.pack(expand=True)

        footer = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        footer.pack(fill="x")
        ttk.Button(footer, text="Restart", command=self.restart_game).pack(
            side="left"
        )
        self.learning_button = ttk.Button(
            footer,
            text="Enable Learning: OFF",
            command=self.toggle_learning,
        )
        self.learning_button.pack(side="left", padx=(8, 0))
        ttk.Label(
            footer,
            text="You are X; the model is O (or vice versa).",
        ).pack(side="right")

    def _show_message(self, message):
        self.status_var.set(message)

    def toggle_learning(self):
        self.learning_enabled = not self.learning_enabled
        self.learning_button.configure(
            text=f"Learning after each game: {'ON' if self.learning_enabled else 'OFF'}"
        )
        if self.learning_enabled:
            self._show_message(
                "Learning is ON. After each completed game, the model will update "
                "from its neural-network-selected moves and save its weights."
            )
        else:
            self._show_message("Learning is OFF. Games will not update model weights.")

    def start_game(self):
        try:
            size = int(self.size_var.get())
            win_count = int(self.win_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid settings",
                "Board size and win condition must be integers.",
            )
            return

        if size < 1 or not 1 <= win_count <= size:
            messagebox.showerror(
                "Invalid settings",
                "Board size must be at least 1, and the win condition must be "
                "between 1 and board size.",
            )
            return

        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Model not found",
                f"Could not find {MODEL_PATH} in the current working directory.",
            )
            return

        try:
            model = GameCNN()
            model(tf.zeros((1, size, size, 3), dtype=tf.float32))
            model.load_weights(MODEL_PATH)
        except Exception as exc:
            messagebox.showerror("Could not load model", str(exc))
            return

        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=ONLINE_LEARNING_RATE
        )
        self.size = size
        self.win_count = win_count
        self.game = Game(size=size, win_count=win_count)
        self.human_player = 1 if self.first_var.get() == "human" else -1
        self.model_player = -self.human_player
        self.game_over = False
        self.model_thinking = False
        self.model_trajectory = []
        self.turn_number = 0

        self._create_board()
        self._refresh_board()

        if self.game.current_player == self.model_player:
            self._show_message("Model is thinking...")
            self.root.after(150, self._model_turn)
        else:
            self._show_message("Your turn. Click an empty square.")

    def _create_board(self):
        for widget in self.board_frame.winfo_children():
            widget.destroy()

        self.cell_buttons = []
        cell_size = max(36, min(78, 520 // self.size))
        font_size = max(12, min(28, cell_size // 2))

        for row in range(self.size):
            button_row = []
            for col in range(self.size):
                button = tk.Button(
                    self.board_frame,
                    text="",
                    width=max(2, cell_size // 14),
                    height=max(1, cell_size // 28),
                    font=("Arial", font_size, "bold"),
                    relief="raised",
                    bd=2,
                    command=lambda r=row, c=col: self._human_move(r, c),
                )
                button.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
                button_row.append(button)
            self.cell_buttons.append(button_row)

        for i in range(self.size):
            self.board_frame.grid_rowconfigure(i, weight=1)
            self.board_frame.grid_columnconfigure(i, weight=1)

    def _refresh_board(self):
        if self.game is None:
            return

        for row in range(self.size):
            for col in range(self.size):
                value = self.game.board[row, col]
                button = self.cell_buttons[row][col]

                if value == 1:
                    text, bg, fg = "X", "#cfe8ff", "#123c66"
                elif value == -1:
                    text, bg, fg = "O", "#ffd9d9", "#762020"
                else:
                    text, bg, fg = "", "#f2f2f2", "#222222"

                button.configure(text=text, bg=bg, fg=fg, activebackground=bg)
                can_click = (
                    not self.game_over
                    and not self.model_thinking
                    and self.game.current_player == self.human_player
                    and value == 0
                )
                button.configure(state="normal" if can_click else "disabled")

    def _human_move(self, row, col):
        if self.game is None or self.game_over or self.model_thinking:
            return
        if self.game.current_player != self.human_player:
            return
        if self.game.board[row, col] != 0:
            return

        self.game.step(row * self.size + col)
        self.turn_number += 1
        self._refresh_board()

        if self._check_game_over():
            return

        self.model_thinking = True
        self._refresh_board()
        self._show_message("Model is thinking...")
        self.root.after(150, self._model_turn)

    def _model_turn(self):
        if self.game is None or self.game_over:
            return

        if self.game.done:
            self._check_game_over()
            return

        try:
            move, move_source, state, legal_moves = select_move(
                self.model, self.game, self.model_player
            )
        except Exception as exc:
            self.model_thinking = False
            self._refresh_board()
            messagebox.showerror("Move selection error", str(exc))
            self._show_message("Move selection failed.")
            return

        if move is None:
            self.model_thinking = False
            self._check_game_over()
            return

        # Record only actions actually selected by the CNN. Forced tactical
        # moves are handled by the override and are not treated as CNN choices.
        if move_source == "Neural network":
            self.model_trajectory.append(
                {
                    "state": state.copy(),
                    "action": int(move),
                    "turn_number": self.turn_number,
                    "legal_moves": legal_moves.copy(),
                }
            )

        self.game.step(move)
        self.turn_number += 1
        self.model_thinking = False
        row, col = divmod(move, self.size)
        self._refresh_board()

        if self._check_game_over():
            return

        self._show_message(
            f"Your turn. Model played row {row + 1}, column {col + 1} ({move_source})."
        )

    def _check_game_over(self):
        if self.game is None or not self.game.done:
            return False

        self.game_over = True
        self.model_thinking = False
        self._refresh_board()

        if self.game.winner == self.human_player:
            message = "You win!"
            outcome = -1.0
        elif self.game.winner == self.model_player:
            message = "The model wins."
            outcome = 1.0
        else:
            message = "Draw."
            outcome = 0.0

        training_message = ""
        if self.learning_enabled:
            try:
                result = train_from_game(
                    self.model,
                    self.optimizer,
                    self.model_trajectory,
                    outcome,
                    self.turn_number,
                )
                if result is None:
                    training_message = (
                        "\n\nNo CNN-selected moves were available to train on "
                        "(the model may have used only forced tactical moves)."
                    )
                else:
                    # Save after every completed game so learning persists.
                    self.model.save_weights(MODEL_PATH)
                    training_message = (
                        f"\n\nModel updated and saved to {MODEL_PATH}."
                        f"\nCNN moves trained: {result['trained_moves']}"
                        f"\nActor loss: {result['actor_loss']:.4f}"
                        f"\nCritic loss: {result['critic_loss']:.4f}"
                    )
            except Exception as exc:
                training_message = f"\n\nTraining failed: {exc}"

        self._show_message(message + training_message.replace("\n", " "))
        messagebox.showinfo("Game over", message + training_message)
        return True

    def restart_game(self):
        if self.game is None:
            self.start_game()
            return

        self.game = Game(size=self.size, win_count=self.win_count)
        self.game_over = False
        self.model_thinking = False
        self.model_trajectory = []
        self.turn_number = 0
        self._refresh_board()

        if self.game.current_player == self.model_player:
            self._show_message("Model is thinking...")
            self.root.after(150, self._model_turn)
        else:
            self._show_message("Your turn. Click an empty square.")


def main():
    root = tk.Tk()
    FourInARowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
