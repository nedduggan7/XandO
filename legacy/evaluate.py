
import csv
import os
import random

import numpy as np
import tensorflow as tf

from game import Game
from game_model import GameCNN


# ============================================================
# CONFIGURATION
# ============================================================

BOARD_SIZE = 5
WIN_COUNT = 4

MODEL_PATH = "actor_critic.weights.h5"
RESULTS_PATH = "evaluation_results.csv"

# Number of games against each opponent.
GAMES_PER_START = 100

# Options: "random" or "tactical"
OPPONENT_TYPE = input(
    "Choose opponent (random/tactical): "
).strip().lower()

if OPPONENT_TYPE not in ("random", "tactical"):
    raise ValueError("Choose either 'random' or 'tactical'.")

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# BOARD ENCODING
# Must match the training script.
# ============================================================

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


# ============================================================
# MODEL MOVE SELECTION
# Selects the highest-scoring legal move.
# No exploration or weight updates during evaluation.
# ============================================================

def select_model_move(model, game, player):
    state = encode_board(
        game.board,
        player,
        game.win_count
    )

    logits, _ = model(
        state[np.newaxis, ...],
        training=False
    )

    scores = logits[0].numpy().reshape(-1)

    legal_moves = game.get_legal_moves()

    if not legal_moves:
        raise RuntimeError("No legal moves available.")

    best_move = max(
        legal_moves,
        key=lambda move: scores[move]
    )

    return int(best_move)


# ============================================================
# TACTICAL OPPONENT
# 1. Take an immediate win.
# 2. Block an immediate opponent win.
# 3. Otherwise choose randomly.
# ============================================================

def find_winning_move(game, player):
    for move in game.get_legal_moves():
        row, col = divmod(move, game.size)

        game.board[row, col] = player
        wins = game.check_win(row, col, player)
        game.board[row, col] = 0

        if wins:
            return move

    return None


def select_opponent_move(game, opponent_type):
    legal_moves = game.get_legal_moves()

    if opponent_type == "random":
        return random.choice(legal_moves)

    # Take an immediate win if available.
    winning_move = find_winning_move(
        game,
        game.current_player
    )

    if winning_move is not None:
        return winning_move

    # Block the player's immediate winning threat.
    blocking_move = find_winning_move(
        game,
        -game.current_player
    )

    if blocking_move is not None:
        return blocking_move

    # No immediate win or threat: play randomly.
    return random.choice(legal_moves)


# ============================================================
# PLAY ONE GAME
# ============================================================

def play_game(model, opponent_type, model_starts_first):
    game = Game(
        size=BOARD_SIZE,
        win_count=WIN_COUNT
    )

    # Player 1 always takes the first turn in Game.
    # The model is player 1 if it starts first,
    # otherwise it is player -1.
    model_player = 1 if model_starts_first else -1

    while not game.done:
        current_player = game.current_player

        if current_player == model_player:
            move = select_model_move(
                model,
                game,
                current_player
            )
        else:
            move = select_opponent_move(
                game,
                opponent_type
            )

        game.step(move)

    if game.winner == model_player:
        result = "Win"
    elif game.winner is None:
        result = "Draw"
    else:
        result = "Loss"

    return {
        "result": result,
        "model_starts_first": model_starts_first,
        "model_player": model_player,
        "winner": game.winner,
        "moves": int(np.count_nonzero(game.board))
    }


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find {MODEL_PATH}. "
        "Train the model first or update MODEL_PATH."
    )

model = GameCNN()

# Build the network before loading its saved weights.
model(
    tf.zeros(
        (1, BOARD_SIZE, BOARD_SIZE, 3),
        dtype=tf.float32
    )
)

model.load_weights(MODEL_PATH)

print(f"\nLoaded model: {MODEL_PATH}")
print(f"Board: {BOARD_SIZE}x{BOARD_SIZE}")
print(f"Win condition: {WIN_COUNT} in a row")
print(f"Opponent: {OPPONENT_TYPE}")
print(f"Games per starting position: {GAMES_PER_START}")


# ============================================================
# RUN EVALUATION
# Alternate the model's starting position.
# ============================================================

results = []

total_games = GAMES_PER_START * 2

for episode in range(total_games):
    model_starts_first = episode % 2 == 0

    game_result = play_game(
        model,
        OPPONENT_TYPE,
        model_starts_first
    )

    game_result["episode"] = episode + 1
    game_result["opponent"] = OPPONENT_TYPE

    results.append(game_result)

    if (episode + 1) % 20 == 0 or episode + 1 == total_games:
        recent = results[-min(20, len(results)):]

        recent_wins = sum(
            r["result"] == "Win" for r in recent
        )

        recent_losses = sum(
            r["result"] == "Loss" for r in recent
        )

        recent_draws = sum(
            r["result"] == "Draw" for r in recent
        )

        print(
            f"Game {episode + 1}/{total_games} | "
            f"Last {len(recent)}: "
            f"W {recent_wins}, "
            f"L {recent_losses}, "
            f"D {recent_draws}"
        )


# ============================================================
# SAVE INDIVIDUAL GAME RESULTS TO CSV
# ============================================================

fieldnames = [
    "episode",
    "opponent",
    "model_starts_first",
    "model_player",
    "winner",
    "result",
    "moves"
]

with open(RESULTS_PATH, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(title, subset):
    games = len(subset)

    wins = sum(r["result"] == "Win" for r in subset)
    losses = sum(r["result"] == "Loss" for r in subset)
    draws = sum(r["result"] == "Draw" for r in subset)

    win_rate = 100 * wins / games if games else 0
    decisive_games = wins + losses
    decisive_win_rate = (
        100 * wins / decisive_games
        if decisive_games else 0
    )

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Games:                  {games}")
    print(f"Wins:                   {wins}")
    print(f"Losses:                 {losses}")
    print(f"Draws:                  {draws}")
    print(f"Win rate (all games):   {win_rate:.1f}%")
    print(
        f"Win rate (no draws):    "
        f"{decisive_win_rate:.1f}%"
    )


print("\n\n========== EVALUATION RESULTS ==========")

print_summary("ALL GAMES", results)

print_summary(
    "MODEL STARTS FIRST",
    [r for r in results if r["model_starts_first"]]
)

print_summary(
    "MODEL STARTS SECOND",
    [r for r in results if not r["model_starts_first"]]
)

print(f"\nDetailed results saved to: {RESULTS_PATH}")
