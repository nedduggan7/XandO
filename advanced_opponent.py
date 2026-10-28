
from functools import lru_cache

import numpy as np


# Search settings
DEFAULT_DEPTH = 1

WIN_SCORE = 1_000_000
INFINITY = float("inf")


def get_legal_moves(board):
    return np.flatnonzero(board.reshape(-1) == 0).tolist()


@lru_cache(maxsize=None)
def get_line_windows(size, win_count):
    """Precompute every contiguous winning-length line on a board."""
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    windows = []
    by_cell = [[] for _ in range(size * size)]
    for row in range(size):
        for col in range(size):
            for dr, dc in directions:
                end_row = row + (win_count - 1) * dr
                end_col = col + (win_count - 1) * dc
                if not (0 <= end_row < size and 0 <= end_col < size):
                    continue
                window = tuple(
                    (row + i * dr, col + i * dc)
                    for i in range(win_count)
                )
                window_index = len(windows)
                windows.append(window)
                for r, c in window:
                    by_cell[r * size + c].append(window_index)
    return tuple(windows), tuple(tuple(x) for x in by_cell)


def get_candidate_moves(board):
    """
    Prioritise empty squares near existing pieces.
    This reduces the branching factor during search.
    """
    size = board.shape[0]
    occupied = np.argwhere(board != 0)

    if len(occupied) == 0:
        centre = (size // 2) * size + (size // 2)
        return [centre]

    candidates = set()

    for row, col in occupied:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r = row + dr
                c = col + dc

                if (
                    0 <= r < size
                    and 0 <= c < size
                    and board[r, c] == 0
                ):
                    candidates.add(r * size + c)

    if not candidates:
        return get_legal_moves(board)

    return list(candidates)


def check_win(board, row, col, player, win_count):
    """
    Check whether the latest move completed a winning line.
    """
    size = board.shape[0]
    directions = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1)
    ]

    for dr, dc in directions:
        count = 1

        for sign in (1, -1):
            for step in range(1, win_count):
                r = row + sign * dr * step
                c = col + sign * dc * step

                if not (0 <= r < size and 0 <= c < size):
                    break

                if board[r, c] != player:
                    break

                count += 1

        if count >= win_count:
            return True

    return False


def evaluate_board(board, root_player, win_count):
    """
    Heuristic evaluation from root_player's perspective.

    Rewards uncontested lines containing the player's pieces.
    Penalises equivalent lines belonging to the opponent.
    """
    size = board.shape[0]
    score = 0

    windows, _ = get_line_windows(size, win_count)
    for window in windows:
        own_count = sum(board[r, c] == root_player for r, c in window)
        opponent_count = sum(board[r, c] == -root_player for r, c in window)
        score += _line_score(own_count, opponent_count)

    # Slight preference for central squares.
    centre = (size - 1) / 2

    for row in range(size):
        for col in range(size):
            if board[row, col] == root_player:
                distance = abs(row - centre) + abs(col - centre)
                score += max(0, size - distance) * 0.2
            elif board[row, col] == -root_player:
                distance = abs(row - centre) + abs(col - centre)
                score -= max(0, size - distance) * 0.2

    return score


def _line_score(own_count, opponent_count):
    if own_count and opponent_count:
        return 0
    weights = (0, 2, 12, 100, 1000, 10000)
    if own_count:
        return weights[min(own_count, len(weights) - 1)]
    if opponent_count:
        return -weights[min(opponent_count, len(weights) - 1)]
    return 0


def evaluate_after_move(board, previous_score, move, player, root_player, win_count):
    """Update the heuristic using only lines and centre value affected by move."""
    size = board.shape[0]
    windows, by_cell = get_line_windows(size, win_count)
    move_row, move_col = divmod(move, size)
    delta = 0

    for window_index in by_cell[move]:
        window = windows[window_index]
        own_before = sum(
            board[r, c] == root_player and (r, c) != (move_row, move_col)
            for r, c in window
        )
        opponent_before = sum(
            board[r, c] == -root_player and (r, c) != (move_row, move_col)
            for r, c in window
        )
        before = _line_score(own_before, opponent_before)
        own_after = own_before + (player == root_player)
        opponent_after = opponent_before + (player == -root_player)
        delta += _line_score(own_after, opponent_after) - before

    centre = (size - 1) / 2
    distance = abs(move_row - centre) + abs(move_col - centre)
    centre_value = max(0, size - distance) * 0.2
    delta += centre_value if player == root_player else -centre_value
    return previous_score + delta


def ordered_moves(board, player, win_count):
    """
    Try immediate wins first, then moves that block an
    immediate opponent win, followed by central moves.
    """
    moves = get_candidate_moves(board)

    def move_priority(move):
        row, col = divmod(move, board.shape[0])

        board[row, col] = player
        wins = check_win(
            board, row, col, player, win_count
        )
        board[row, col] = 0

        if wins:
            return 1_000_000

        board[row, col] = -player
        blocks = check_win(
            board, row, col, -player, win_count
        )
        board[row, col] = 0

        if blocks:
            return 500_000

        centre = (board.shape[0] - 1) / 2
        distance = abs(row - centre) + abs(col - centre)

        # Prefer central squares when other factors are equal.
        return -distance

    return sorted(
        moves,
        key=move_priority,
        reverse=True
    )


def minimax(
    board,
    player_to_move,
    root_player,
    win_count,
    depth,
    alpha,
    beta,
    last_move=None,
    ply=0,
    eval_score=None,
    transposition_table=None
):
    """
    Minimax with alpha-beta pruning.

    Returns a score from root_player's perspective.
    """
    if last_move is not None:
        row, col, last_player = last_move

        if check_win(board, row, col, last_player, win_count):
            if last_player == root_player:
                return WIN_SCORE - ply
            return -WIN_SCORE + ply

    if not np.any(board == 0):
        return 0

    if depth <= 0:
        if eval_score is None:
            return evaluate_board(board, root_player, win_count)
        return eval_score

    if transposition_table is None:
        transposition_table = {}

    key = (
        board.tobytes(), player_to_move, root_player, win_count, depth
    )
    original_alpha = alpha
    original_beta = beta
    cached = transposition_table.get(key)
    cached_move = None
    if cached is not None:
        cached_score, bound, cached_move = cached
        if bound == "exact":
            return cached_score
        if bound == "lower":
            alpha = max(alpha, cached_score)
        else:  # upper bound
            beta = min(beta, cached_score)
        if alpha >= beta:
            return cached_score

    moves = ordered_moves(
        board,
        player_to_move,
        win_count
    )
    if not moves:
        return 0
    if cached_move in moves:
        moves.remove(cached_move)
        moves.insert(0, cached_move)

    if player_to_move == root_player:
        best_score = -INFINITY
        best_node_move = None

        for move in moves:
            row, col = divmod(move, board.shape[0])
            board[row, col] = player_to_move
            child_eval = evaluate_after_move(
                board, eval_score, move, player_to_move,
                root_player, win_count
            ) if eval_score is not None else None

            score = minimax(
                board,
                -player_to_move,
                root_player,
                win_count,
                depth - 1,
                alpha,
                beta,
                (row, col, player_to_move),
                ply + 1,
                child_eval,
                transposition_table
            )

            board[row, col] = 0

            if score > best_score:
                best_score = score
                best_node_move = move
            alpha = max(alpha, best_score)

            if beta <= alpha:
                break

        result = best_score

    else:
        best_score = INFINITY
        best_node_move = None

        for move in moves:
            row, col = divmod(move, board.shape[0])
            board[row, col] = player_to_move
            child_eval = evaluate_after_move(
                board, eval_score, move, player_to_move,
                root_player, win_count
            ) if eval_score is not None else None

            score = minimax(
                board,
                -player_to_move,
                root_player,
                win_count,
                depth - 1,
                alpha,
                beta,
                (row, col, player_to_move),
                ply + 1,
                child_eval,
                transposition_table
            )

            board[row, col] = 0
            if score < best_score:
                best_score = score
                best_node_move = move
            beta = min(beta, best_score)

            if beta <= alpha:
                break
        result = best_score

    if result <= original_alpha:
        bound = "upper"
    elif result >= original_beta:
        bound = "lower"
    else:
        bound = "exact"
    transposition_table[key] = (result, bound, best_node_move)
    return result


def select_move(game, depth=DEFAULT_DEPTH):
    """
    Select a move for the current player in a Game instance.

    Returns a flattened board index, compatible with game.step().
    """
    if game.done:
        raise ValueError("Cannot choose a move: game is over.")

    board = game.board
    player = game.current_player
    win_count = game.win_count

    moves = ordered_moves(board, player, win_count)

    best_move = None
    best_score = -INFINITY
    alpha = -INFINITY
    beta = INFINITY

    # Keep the tactical move ordering: alpha-beta pruning benefits from it.
    root_eval = evaluate_board(board, player, win_count)
    transposition_table = {}

    for move in moves:
        row, col = divmod(move, board.shape[0])
        board[row, col] = player
        child_eval = evaluate_after_move(
            board, root_eval, move, player, player, win_count
        )

        score = minimax(
            board,
            -player,
            player,
            win_count,
            depth - 1,
            alpha,
            beta,
            (row, col, player),
            ply=1,
            eval_score=child_eval,
            transposition_table=transposition_table
        )

        board[row, col] = 0

        if score > best_score:
            best_score = score
            best_move = move
        elif score == best_score and np.random.random() < 0.5:
            best_move = move

        alpha = max(alpha, best_score)

    return int(best_move)
