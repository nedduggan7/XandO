
import os
import numpy as np
import tensorflow as tf

from game import Game
from game_model import GameCNN


MODEL_PATH = "actor_critic.weights.h5"


def encode_board(board, player, win_count):
    size = board.shape[0]

    own = (board == player).astype(np.float32)
    opponent = (board == -player).astype(np.float32)

    rule = np.full(
        (size, size),
        win_count / size,
        dtype=np.float32
    )

    return np.stack([own, opponent, rule], axis=-1)


def find_winning_moves(game, player):
    """Return legal moves that immediately win for the given player."""
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
    """
    Priority:
    1. Take an immediate win.
    2. Block a single immediate opponent winning move.
    3. Otherwise, defer to the CNN.

    If the opponent has two or more immediate winning moves, one block
    cannot generally prevent defeat, so the CNN chooses instead.
    """
    own_wins = find_winning_moves(game, player)
    if own_wins:
        return own_wins[0]

    opponent_wins = find_winning_moves(game, -player)
    if len(opponent_wins) == 1:
        return opponent_wins[0]

    return None


def select_move(model, game, player):
    # Apply exact tactical rules before consulting the CNN.
    tactical_move = get_tactical_move(game, player)
    if tactical_move is not None:
        return int(tactical_move)

    state = encode_board(
        game.board, player, game.win_count
    )

    # Actor-critic model returns policy logits and value estimate.
    logits, value = model(
        state[np.newaxis, ...],
        training=False
    )

    scores = logits[0].numpy().reshape(-1)

    legal_moves = np.asarray(
        game.get_legal_moves(),
        dtype=np.int32
    )

    if len(legal_moves) == 0:
        return None

    # Select the highest-scoring legal move.
    legal_scores = scores[legal_moves]
    return int(legal_moves[np.argmax(legal_scores)])


def print_board(board):
    size = board.shape[0]

    print("\n   " + " ".join(f"{c + 1:2}" for c in range(size)))
    print("   " + "---" * size)

    for r in range(size):
        symbols = []

        for c in range(size):
            value = board[r, c]

            if value == 1:
                symbols.append(" X")
            elif value == -1:
                symbols.append(" O")
            else:
                symbols.append(" .")

        print(f"{r + 1:2} |" + " ".join(symbols))

    print()


def main():
    try:
        size = int(input("Board size (e.g. 5): "))
        win_count = int(input("Pieces in a row to win (e.g. 4): "))
    except ValueError:
        print("Enter valid integer values.")
        return

    if size < 1 or not 1 <= win_count <= size:
        print("Invalid board size or win condition.")
        return

    game = Game(size=size, win_count=win_count)

    model = GameCNN()

    # Build model variables before loading weights
    model(tf.zeros((1, size, size, 3)))

    if not os.path.exists(MODEL_PATH):
        print(f"Model weights not found: {MODEL_PATH}")
        print("Run train.py first.")
        return

    try:
        model.load_weights(MODEL_PATH)
    except Exception as exc:
        print(f"Could not load model weights: {exc}")
        return

    print("Actor-critic model loaded.")

    choice = input("Play first? (y/n): ").strip().lower()
    human_player = 1 if choice == "y" else -1
    model_player = -human_player

    print("You are X. The model is O." if human_player == 1
          else "You are O. The model is X.")

    print_board(game.board)

    while not game.done:

        if game.current_player == human_player:
            print("Your turn.")

            try:
                row = int(input(f"Row (1-{size}): ")) - 1
                col = int(input(f"Column (1-{size}): ")) - 1
            except ValueError:
                print("Enter valid integer coordinates.")
                continue

            if not (0 <= row < size and 0 <= col < size):
                print("Position is outside the board.")
                continue

            move = row * size + col

            if move not in game.get_legal_moves():
                print("That square is occupied.")
                continue

        else:
            print("Model is thinking...")
            move = select_move(model, game, model_player)

            if move is None:
                break

            row, col = divmod(move, size)
            print(f"Model chooses row {row + 1}, column {col + 1}.")

        game.step(move)
        print_board(game.board)

    if game.winner == human_player:
        print("You win!")
    elif game.winner == model_player:
        print("The model wins.")
    else:
        print("Draw.")


if __name__ == "__main__":
    main()
