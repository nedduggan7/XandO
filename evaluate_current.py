"""Evaluate the saved 6x6 model against its training opponents."""

import csv
import random

import numpy as np
import tensorflow as tf

from advanced_opponent import select_move as select_advanced_move
from game import Game
from game_model import GameCNN


BOARD_SIZE = 6
WIN_COUNT = 4
MODEL_PATH = "actor_critic.weights.h5"
GAMES_PER_START = 5
ADVANCED_DEPTH = 4
RESULTS_PATH = "evaluation_current_results.csv"
SEED = 20261028

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def encode_board(board, player):
    own = (board == player).astype(np.float32)
    opponent = (board == -player).astype(np.float32)
    rule = np.full((BOARD_SIZE, BOARD_SIZE), WIN_COUNT / BOARD_SIZE,
                   dtype=np.float32)
    return np.stack([own, opponent, rule], axis=-1)


def winning_moves(game, player):
    wins = []
    for move in game.get_legal_moves():
        row, col = divmod(move, BOARD_SIZE)
        game.board[row, col] = player
        if game.check_win(row, col, player):
            wins.append(move)
        game.board[row, col] = 0
    return wins


def open_three_templates():
    templates = []
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            for dr, dc in directions:
                end = (row + 2 * dr, col + 2 * dc)
                before = (row - dr, col - dc)
                after = (end[0] + dr, end[1] + dc)
                if not all(0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
                           for r, c in (before, end, after)):
                    continue
                cells = tuple((row + i * dr, col + i * dc) for i in range(3))
                templates.append((cells, before, after))
    return templates


OPEN_THREE_TEMPLATES = open_three_templates()


def open_three_setup_moves(game, player, blocked_move=None):
    board = game.board
    setup_moves = set()
    for cells, before, after in OPEN_THREE_TEMPLATES:
        if (board[before] != 0 or board[after] != 0
                or blocked_move in (before, after)):
            continue
        for row, col in cells:
            move = row * BOARD_SIZE + col
            if board[row, col] != 0 or blocked_move == (row, col):
                continue
            if all((r, c) == (row, col) or board[r, c] == player
                   for r, c in cells):
                setup_moves.add(move)
    return sorted(setup_moves)


def tactical_moves(game, player):
    own_wins = winning_moves(game, player)
    if own_wins:
        return own_wins

    opponent_wins = winning_moves(game, -player)
    if opponent_wins:
        best, minimum = [], None
        for move in game.get_legal_moves():
            row, col = divmod(move, BOARD_SIZE)
            game.board[row, col] = player
            remaining = len(winning_moves(game, -player))
            game.board[row, col] = 0
            if minimum is None or remaining < minimum:
                best, minimum = [move], remaining
            elif remaining == minimum:
                best.append(move)
        return best or opponent_wins

    own_setups = open_three_setup_moves(game, player)
    if own_setups:
        return own_setups

    opponent_setups = open_three_setup_moves(game, -player)
    if not opponent_setups:
        return None

    best, minimum = [], None
    for move in game.get_legal_moves():
        row, col = divmod(move, BOARD_SIZE)
        remaining = len(open_three_setup_moves(
            game, -player, blocked_move=(row, col)
        ))
        if minimum is None or remaining < minimum:
            best, minimum = [move], remaining
        elif remaining == minimum:
            best.append(move)
    return best or opponent_setups


def model_move(model, game, player):
    state = encode_board(game.board, player)
    logits, _ = model(state[np.newaxis, ...], training=False)
    scores = logits[0].numpy().reshape(-1)
    forced = tactical_moves(game, player)
    choices = forced if forced is not None else game.get_legal_moves()
    return int(choices[int(np.argmax(scores[choices]))])


def tactical_opponent_move(game):
    forced = tactical_moves(game, game.current_player)
    if forced:
        return forced[0]
    return random.choice(game.get_legal_moves())


def play_game(model, opponent, model_starts_first):
    game = Game(size=BOARD_SIZE, win_count=WIN_COUNT)
    model_player = 1 if model_starts_first else -1
    while not game.done:
        player = game.current_player
        if opponent == "self_play" or player == model_player:
            move = model_move(model, game, player)
        elif opponent == "advanced":
            move = select_advanced_move(game, depth=ADVANCED_DEPTH)
        elif opponent == "tactical":
            move = tactical_opponent_move(game)
        else:
            move = random.choice(game.get_legal_moves())
        game.step(move)

    if game.winner is None:
        result = "Draw"
    elif game.winner == model_player:
        result = "Win"
    else:
        result = "Loss"
    return result, int(np.count_nonzero(game.board))


def main():
    model = GameCNN()
    model(tf.zeros((1, BOARD_SIZE, BOARD_SIZE, 3), dtype=tf.float32))
    model.load_weights(MODEL_PATH)
    print(f"Loaded {MODEL_PATH}; board {BOARD_SIZE}x{BOARD_SIZE}, win {WIN_COUNT}.",
          flush=True)

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=("opponent", "model_starts_first", "result", "moves"),
        )
        writer.writeheader()
        result_file.flush()

        for opponent in ("advanced", "tactical", "random", "self_play"):
            counts = {"Win": 0, "Loss": 0, "Draw": 0}
            moves = []
            for starts_first in (True, False):
                for game_number in range(1, GAMES_PER_START + 1):
                    result, length = play_game(model, opponent, starts_first)
                    counts[result] += 1
                    moves.append(length)
                    writer.writerow({
                        "opponent": opponent,
                        "model_starts_first": starts_first,
                        "result": result,
                        "moves": length,
                    })
                    result_file.flush()
                    print(
                        f"{opponent}: {'first' if starts_first else 'second'} "
                        f"{game_number}/{GAMES_PER_START}: {result} ({length} moves)",
                        flush=True,
                    )
            total = sum(counts.values())
            score_rate = 100 * (counts["Win"] + 0.5 * counts["Draw"]) / total
            print(
                f"SUMMARY {opponent}: {counts['Win']}W "
                f"{counts['Loss']}L {counts['Draw']}D; "
                f"score rate {score_rate:.1f}%; mean game length "
                f"{np.mean(moves):.1f}",
                flush=True,
            )
    print(f"Per-game results saved to {RESULTS_PATH}.", flush=True)


if __name__ == "__main__":
    main()
