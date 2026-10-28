import os
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import tensorflow as tf

from game import Game
from game_model import GameCNN


MODEL_PATH = "actor_critic.weights.h5"


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
    tactical_move = get_tactical_move(game, player)
    if tactical_move is not None:
        return int(tactical_move), "Tactical move"

    state = encode_board(game.board, player, game.win_count)
    logits, _ = model(state[np.newaxis, ...], training=False)
    scores = logits[0].numpy().reshape(-1)

    legal_moves = np.asarray(game.get_legal_moves(), dtype=np.int32)
    if len(legal_moves) == 0:
        return None, "No legal moves"

    legal_scores = scores[legal_moves]
    return int(legal_moves[np.argmax(legal_scores)]), "Neural network"


class FourInARowGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Four in a Row — Neural Network")
        self.root.minsize(420, 460)

        self.game = None
        self.model = None
        self.human_player = 1
        self.model_player = -1
        self.cell_buttons = []
        self.game_over = True
        self.model_thinking = False

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
            row=1, column=2, columnspan=2, sticky="ew", padx=(0, 0), pady=(9, 0)
        )

        self.status_var = tk.StringVar(value="")
        ttk.Label(
            self.root,
            textvariable=self.status_var,
            padding=(10, 4),
            anchor="center",
            wraplength=620,
        ).pack(fill="x")

        self.board_frame = ttk.Frame(self.root, padding=(10, 4, 10, 10))
        self.board_frame.pack(expand=True)

        footer = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        footer.pack(fill="x")
        ttk.Button(footer, text="Restart", command=self.restart_game).pack(
            side="left"
        )
        ttk.Label(
            footer,
            text="You are X; the model is O (or vice versa).",
        ).pack(side="right")

    def _show_message(self, message):
        self.status_var.set(message)

    def start_game(self):
        try:
            size = int(self.size_var.get())
            win_count = int(self.win_var.get())
        except ValueError:
            messagebox.showerror("Invalid settings", "Board size and win condition must be integers.")
            return

        if size < 1 or not 1 <= win_count <= size:
            messagebox.showerror(
                "Invalid settings",
                "Board size must be at least 1, and the win condition must be between 1 and board size.",
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
        self.size = size
        self.win_count = win_count
        self.game = Game(size=size, win_count=win_count)
        self.human_player = 1 if self.first_var.get() == "human" else -1
        self.model_player = -self.human_player
        self.game_over = False
        self.model_thinking = False

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
        # Scale the cells down for larger boards to keep the window manageable.
        cell_size = max(36, min(78, 520 // self.size))
        font_size = max(12, min(28, cell_size // 2))

        for row in range(self.size):
            button_row = []
            for col in range(self.size):
                button = tk.Button(
                    self.board_frame,
                    text="",
                    width=2,
                    height=1,
                    font=("Arial", font_size, "bold"),
                    relief="raised",
                    bd=2,
                    command=lambda r=row, c=col: self._human_move(r, c),
                )
                button.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
                button.configure(width=max(2, cell_size // 14), height=max(1, cell_size // 28))
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
            move, move_source = select_move(self.model, self.game, self.model_player)
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

        self.game.step(move)
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
        elif self.game.winner == self.model_player:
            message = "The model wins."
        else:
            message = "Draw."

        self._show_message(message)
        messagebox.showinfo("Game over", message)
        return True

    def restart_game(self):
        if self.game is None:
            self.start_game()
            return

        self.game = Game(size=self.size, win_count=self.win_count)
        self.game_over = False
        self.model_thinking = False
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
