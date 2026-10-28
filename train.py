
import os
import random
from functools import lru_cache

import numpy as np
import tensorflow as tf

from game import Game
from game_model import GameCNN
from advanced_opponent import select_move as select_advanced_move


# ============================================================
# CONFIGURATION
# ============================================================

BOARD_SIZE = 6
WIN_COUNT = 4

EPISODES = 1000
PRINT_EVERY = 100
EVALUATE_EVERY = 500
SAVE_EVERY = 500

LEARNING_RATE = 0.0003
GAMMA = 0.99

VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
TACTICAL_COEF = 1.0

GRADIENT_CLIP_NORM = 1.0

# Begin with a shallower search opponent.
ADVANCED_DEPTH = 4

# Opponent distribution (weights are relative percentages).
OPPONENT_TYPES = [
    "advanced",
    "tactical",
    "random",
    "self_play"
]

OPPONENT_WEIGHTS = [30, 50, 10, 10]
ADVANCED_START_WEIGHT = 10
ADVANCED_FINAL_WEIGHT = OPPONENT_WEIGHTS[0]

MODEL_PATH = "actor_critic.weights.h5"
CHECKPOINT_DIR = "checkpoints"

EVAL_GAMES_PER_START = 5

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# BOARD ENCODING
# Channel 1: current player's pieces.
# Channel 2: opponent's pieces.
# Channel 3: win condition.
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
# FIND IMMEDIATE WINNING MOVES
# ============================================================

def find_winning_moves(game, player):
    """
    Return every legal move that immediately wins for player.
    The real board is restored after each simulated move.
    """
    winning_moves = []

    for move in game.get_legal_moves():
        row, col = divmod(move, game.size)

        game.board[row, col] = player

        wins = game.check_win(
            row,
            col,
            player
        )

        game.board[row, col] = 0

        if wins:
            winning_moves.append(move)

    return winning_moves


# ============================================================
# TACTICAL OVERRIDE
# ============================================================

@lru_cache(maxsize=None)
def get_open_run_templates(size, target_length):
    """Precompute every run of target_length with an in-bounds open end."""
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    templates = []
    for row in range(size):
        for col in range(size):
            for dr, dc in directions:
                end_row = row + (target_length - 1) * dr
                end_col = col + (target_length - 1) * dc
                before = (row - dr, col - dc)
                after = (end_row + dr, end_col + dc)
                if not (
                    0 <= before[0] < size and 0 <= before[1] < size
                    and 0 <= end_row < size and 0 <= end_col < size
                    and 0 <= after[0] < size and 0 <= after[1] < size
                ):
                    continue
                cells = tuple(
                    (row + i * dr, col + i * dc)
                    for i in range(target_length)
                )
                templates.append((cells, before, after))
    return tuple(templates)


def find_open_three_setup_moves(
    game, player, blocked_move=None
):
    """Find moves that create an open run, without simulating board edits.

    blocked_move optionally treats one empty square as occupied by the
    defending player, for fast evaluation of a candidate block.
    """
    target_length = game.win_count - 1
    if target_length < 1:
        return []
    board = game.board
    setup_moves = set()
    for cells, before, after in get_open_run_templates(
        game.size, target_length
    ):
        if (
            board[before] != 0
            or board[after] != 0
            or blocked_move in (before, after)
        ):
            continue
        for setup_row, setup_col in cells:
            setup_move = setup_row * game.size + setup_col
            if board[setup_row, setup_col] != 0:
                continue
            if blocked_move == (setup_row, setup_col):
                continue
            if all(
                (r, c) == (setup_row, setup_col)
                or board[r, c] == player
                for r, c in cells
            ):
                setup_moves.add(setup_move)
    return sorted(setup_moves)


def opponent_weights_for_episode(episode):
    """Increase advanced-opponent exposure from 10% to its 30% target."""
    progress = (episode - 1) / max(EPISODES - 1, 1)
    advanced_weight = (
        ADVANCED_START_WEIGHT
        + progress * (ADVANCED_FINAL_WEIGHT - ADVANCED_START_WEIGHT)
    )
    other_total = sum(OPPONENT_WEIGHTS[1:])
    remaining = 100 - advanced_weight
    weights = [advanced_weight]
    weights.extend(
        remaining * weight / other_total
        for weight in OPPONENT_WEIGHTS[1:]
    )
    return weights


def get_tactical_moves(game, player):
    """Return the hard-required move set for this position, or None.

    Priority: win now; prevent an immediate opponent win; create our own
    open (win_count - 1) run; then prevent the opponent creating one. If a
    defensive move cannot remove every threat, retain the moves that leave
    the fewest threats. The model ranks the remaining allowed moves.
    """
    own_wins = find_winning_moves(game, player)
    if own_wins:
        return own_wins

    opponent_wins = find_winning_moves(game, -player)
    if opponent_wins:
        legal_moves = game.get_legal_moves()
        best_moves = []
        fewest_remaining = None
        for move in legal_moves:
            row, col = divmod(move, game.size)
            game.board[row, col] = player
            remaining = len(find_winning_moves(game, -player))
            game.board[row, col] = 0
            if fewest_remaining is None or remaining < fewest_remaining:
                fewest_remaining = remaining
                best_moves = [move]
            elif remaining == fewest_remaining:
                best_moves.append(move)
        return best_moves or opponent_wins

    own_setup_moves = find_open_three_setup_moves(game, player)
    if own_setup_moves:
        return own_setup_moves

    opponent_setup_moves = find_open_three_setup_moves(game, -player)
    if not opponent_setup_moves:
        return None

    # Occupying a setup square prevents that specific fork. If there are
    # several, prefer moves that eliminate the most opponent setup options.
    legal_moves = game.get_legal_moves()
    best_moves = []
    fewest_remaining = None
    for move in legal_moves:
        row, col = divmod(move, game.size)
        remaining = len(find_open_three_setup_moves(
            game, -player, blocked_move=(row, col)
        ))
        if fewest_remaining is None or remaining < fewest_remaining:
            fewest_remaining = remaining
            best_moves = [move]
        elif remaining == fewest_remaining:
            best_moves.append(move)
    return best_moves or opponent_setup_moves


def get_tactical_target(game, player):
    """Return the first hard-required move (used by the basic opponent)."""
    moves = get_tactical_moves(game, player)
    return moves[0] if moves else None


# ============================================================
# MODEL MOVE SELECTION
# ============================================================

def select_model_move(model, game, player, stochastic=True):
    """
    Tactical override first; otherwise use the CNN.

    Returns:
        move: selected flattened square index
        state: encoded board from this player's perspective
        was_forced: True if the tactical layer selected the move
    """
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
    # Tactical rules constrain the legal action set. The policy still ranks
    # all allowed winning/blocking squares using its learned logits.
    tactical_moves = get_tactical_moves(game, player)
    legal_moves = (
        tactical_moves if tactical_moves is not None
        else game.get_legal_moves()
    )

    if not legal_moves:
        raise RuntimeError("No legal moves available.")

    legal_scores = scores[legal_moves]

    if not stochastic:
        index = int(np.argmax(legal_scores))
        return int(legal_moves[index]), state, tactical_moves is not None

    # Sample only legal moves during training.
    legal_scores = legal_scores - np.max(legal_scores)

    probabilities = np.exp(legal_scores)
    probabilities /= probabilities.sum()

    move = int(
        np.random.choice(
            legal_moves,
            p=probabilities
        )
    )

    return move, state, tactical_moves is not None


# ============================================================
# BASIC TACTICAL OPPONENT
# ============================================================

def select_tactical_move(game):
    # Use the same immediate-win and threat-minimisation logic as the model.
    target = get_tactical_target(game, game.current_player)
    if target is not None:
        return target
    return random.choice(game.get_legal_moves())


# ============================================================
# OPPONENT MOVE SELECTION
# ============================================================

def select_opponent_move(game, opponent_type):
    if opponent_type == "advanced":
        return select_advanced_move(
            game,
            depth=ADVANCED_DEPTH
        )

    if opponent_type == "tactical":
        return select_tactical_move(game)

    if opponent_type == "random":
        return random.choice(game.get_legal_moves())

    raise ValueError(
        f"Unknown opponent type: {opponent_type}"
    )


# ============================================================
# PLAY ONE TRAINING GAME
# ============================================================

def play_training_game(
    model,
    opponent_type,
    model_starts_first
):
    game = Game(
        size=BOARD_SIZE,
        win_count=WIN_COUNT
    )

    model_player = 1 if model_starts_first else -1

    # Only CNN-selected decisions are recorded for actor/critic learning.
    # Tactical overrides are executed but are not treated as CNN actions.
    trajectories = {
        1: [],
        -1: []
    }
    tactical_examples = []

    turn_number = 0

    while not game.done:
        player = game.current_player

        model_controls_turn = (
            opponent_type == "self_play"
            or player == model_player
        )

        if model_controls_turn:
            move, state, was_forced = select_model_move(
                model,
                game,
                player,
                stochastic=True
            )

            if was_forced:
                tactical_examples.append((state, move))
            else:
                trajectories[player].append(
                    (state, move, turn_number)
                )
        else:
            move = select_opponent_move(
                game,
                opponent_type
            )

        game.step(move)
        turn_number += 1

    # Terminal reward from each player's perspective.
    rewards = {}

    for player in (1, -1):
        if game.winner is None:
            rewards[player] = 0.0
        elif game.winner == player:
            rewards[player] = 1.0
        else:
            rewards[player] = -1.0

    return (
        trajectories,
        rewards,
        game.winner,
        turn_number,
        tactical_examples
    )


# ============================================================
# ACTOR-CRITIC UPDATE
# ============================================================

def train_on_game(
    model,
    optimizer,
    trajectories,
    rewards,
    total_turns,
    tactical_examples=None
):
    if tactical_examples is None:
        tactical_examples = []
    states_list = []
    actions_list = []
    returns_list = []

    # Only decisions made by the CNN (not tactical overrides) contribute
    # to the policy-gradient and critic losses.
    for player in (1, -1):
        for state, action, turn_number in trajectories[player]:
            discounted_return = rewards[player] * (
                GAMMA ** (total_turns - turn_number - 1)
            )

            states_list.append(state)
            actions_list.append(action)
            returns_list.append(discounted_return)

    # Add tactical examples as supervised policy targets. This allows the
    # CNN to learn winning/blocking decisions even though the tactical layer
    # selected those moves during gameplay.
    if not states_list and not tactical_examples:
        return None

    policy_states = [state for state, _ in tactical_examples]
    policy_actions = [move for _, move in tactical_examples]

    states = np.stack(states_list).astype(np.float32) if states_list else None
    actions = np.asarray(actions_list, dtype=np.int32)
    returns = tf.convert_to_tensor(returns_list, dtype=tf.float32)

    if policy_states:
        tactical_states = np.stack(policy_states).astype(np.float32)
        tactical_actions = np.asarray(policy_actions, dtype=np.int32)
    else:
        tactical_states = None
        tactical_actions = np.asarray([], dtype=np.int32)

    number_of_squares = BOARD_SIZE * BOARD_SIZE

    with tf.GradientTape() as tape:
        actor_loss = tf.constant(0.0, dtype=tf.float32)
        critic_loss = tf.constant(0.0, dtype=tf.float32)
        entropy = tf.constant(0.0, dtype=tf.float32)
        tactical_loss = tf.constant(0.0, dtype=tf.float32)
        total_loss = tf.constant(0.0, dtype=tf.float32)

        if states is not None:
            batch_size = len(states)
            occupied = (
                states[:, :, :, 0] + states[:, :, :, 1]
            ).reshape(batch_size, -1) > 0.5
            legal_mask = tf.convert_to_tensor(~occupied)

            logits, values = model(states, training=True)
            logits = tf.reshape(logits, (batch_size, number_of_squares))
            logits = tf.where(legal_mask, logits, tf.cast(-1e9, logits.dtype))
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            probabilities = tf.nn.softmax(logits, axis=-1)
            selected_log_probs = tf.gather(log_probs, actions, axis=1, batch_dims=1)
            advantages = tf.stop_gradient(returns - values)
            actor_loss = -tf.reduce_mean(selected_log_probs * advantages)
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            entropy = -tf.reduce_mean(
                tf.reduce_sum(probabilities * log_probs, axis=-1)
            )
            total_loss += actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

        if tactical_states is not None:
            tactical_logits, _ = model(tactical_states, training=True)
            tactical_batch = len(tactical_states)
            tactical_logits = tf.reshape(
                tactical_logits, (tactical_batch, number_of_squares)
            )
            tactical_occupied = (
                tactical_states[:, :, :, 0] + tactical_states[:, :, :, 1]
            ).reshape(tactical_batch, -1) > 0.5
            tactical_legal_mask = tf.convert_to_tensor(~tactical_occupied)
            tactical_logits = tf.where(
                tactical_legal_mask,
                tactical_logits,
                tf.cast(-1e9, tactical_logits.dtype)
            )
            tactical_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tactical_actions,
                    logits=tactical_logits
                )
            )
            total_loss += TACTICAL_COEF * tactical_loss

    gradients = tape.gradient(
        total_loss,
        model.trainable_variables
    )

    gradient_pairs = [
        (gradient, variable)
        for gradient, variable in zip(
            gradients,
            model.trainable_variables
        )
        if gradient is not None
    ]

    if gradient_pairs:
        grads, variables = zip(*gradient_pairs)

        grads, _ = tf.clip_by_global_norm(
            grads,
            GRADIENT_CLIP_NORM
        )

        optimizer.apply_gradients(
            zip(grads, variables)
        )

    return {
        "actor_loss": float(actor_loss.numpy()),
        "critic_loss": float(critic_loss.numpy()),
        "entropy": float(entropy.numpy()),
        "tactical_loss": float(tactical_loss.numpy()),
        "total_loss": float(total_loss.numpy()),
        "cnn_decisions": len(states_list),
        "tactical_examples": len(tactical_examples)
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_against(model, opponent_type):
    results = {
        "first": {"Win": 0, "Loss": 0, "Draw": 0},
        "second": {"Win": 0, "Loss": 0, "Draw": 0}
    }

    for starts_first in (True, False):
        key = "first" if starts_first else "second"
        model_player = 1 if starts_first else -1

        for _ in range(EVAL_GAMES_PER_START):
            game = Game(
                size=BOARD_SIZE,
                win_count=WIN_COUNT
            )

            while not game.done:
                player = game.current_player

                if player == model_player:
                    move, _, _ = select_model_move(
                        model,
                        game,
                        player,
                        stochastic=False
                    )
                else:
                    move = select_opponent_move(
                        game,
                        opponent_type
                    )

                game.step(move)

            if game.winner is None:
                result = "Draw"
            elif game.winner == model_player:
                result = "Win"
            else:
                result = "Loss"

            results[key][result] += 1

    return results


def print_evaluation(opponent_type, results):
    print(
        f"\nEvaluation against {opponent_type.upper()}"
    )

    for position, stats in results.items():
        total = sum(stats.values())

        win_rate = 100 * stats["Win"] / total

        score_rate = (
            100
            * (stats["Win"] + 0.5 * stats["Draw"])
            / total
        )

        print(
            f"  Model starts {position:6s}: "
            f"W {stats['Win']:2d} | "
            f"L {stats['Loss']:2d} | "
            f"D {stats['Draw']:2d} | "
            f"Win rate {win_rate:5.1f}% | "
            f"Score rate {score_rate:5.1f}%"
        )


# ============================================================
# INITIALISE MODEL
# ============================================================

model = GameCNN()

# Build the model before loading weights.
model(
    tf.zeros(
        (1, BOARD_SIZE, BOARD_SIZE, 3),
        dtype=tf.float32
    )
)

if os.path.exists(MODEL_PATH):
    model.load_weights(MODEL_PATH)
    print(f"Loaded existing model: {MODEL_PATH}")
else:
    print("No existing weights found. Starting from random weights.")

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


# ============================================================
# TRAINING LOOP
# ============================================================

start_counters = {
    name: 0
    for name in OPPONENT_TYPES
}

recent_results = {
    name: []
    for name in OPPONENT_TYPES
    if name != "self_play"
}

recent_losses = []

print("\nStarting mixed training with tactical move overrides.")
print(
    "Advanced opponent share ramps from "
    f"{ADVANCED_START_WEIGHT}% to {ADVANCED_FINAL_WEIGHT}%"
)
print(f"Advanced opponent depth: {ADVANCED_DEPTH}")

for episode in range(1, EPISODES + 1):

    opponent_type = random.choices(
        OPPONENT_TYPES,
        weights=opponent_weights_for_episode(episode),
        k=1
    )[0]

    # Alternate model starting position for each opponent type.
    model_starts_first = (
        start_counters[opponent_type] % 2 == 0
    )

    start_counters[opponent_type] += 1

    (
        trajectories,
        rewards,
        winner,
        total_turns,
        tactical_examples
    ) = play_training_game(
        model,
        opponent_type,
        model_starts_first
    )

    losses = train_on_game(
        model,
        optimizer,
        trajectories,
        rewards,
        total_turns,
        tactical_examples
    )

    # Track results only against actual opponents.
    if opponent_type != "self_play":
        model_player = (
            1 if model_starts_first else -1
        )

        if winner is None:
            result = "Draw"
        elif winner == model_player:
            result = "Win"
        else:
            result = "Loss"

        recent_results[opponent_type].append(
            result
        )

        if len(recent_results[opponent_type]) > 100:
            recent_results[opponent_type].pop(0)

    if losses is not None:
        recent_losses.append(losses)

        if len(recent_losses) > 100:
            recent_losses.pop(0)

    # --------------------------------------------------------
    # PRINT PROGRESS
    # --------------------------------------------------------

    if episode % PRINT_EVERY == 0:
        print(f"\nEpisode {episode}/{EPISODES}")
        current_weights = opponent_weights_for_episode(episode)
        print(
            "  Opponent mix: "
            + " | ".join(
                f"{name} {weight:.1f}%"
                for name, weight in zip(OPPONENT_TYPES, current_weights)
            )
        )

        for opponent, recent in recent_results.items():
            if not recent:
                continue

            print(
                f"  {opponent:8s} last {len(recent):3d}: "
                f"W {recent.count('Win'):3d} | "
                f"L {recent.count('Loss'):3d} | "
                f"D {recent.count('Draw'):3d}"
            )

        if recent_losses:
            print(
                "  Mean actor loss: "
                f"{np.mean([x['actor_loss'] for x in recent_losses]):.4f}"
            )

            print(
                "  Mean critic loss: "
                f"{np.mean([x['critic_loss'] for x in recent_losses]):.4f}"
            )
            print(
                "  Mean tactical loss: "
                f"{np.mean([x['tactical_loss'] for x in recent_losses]):.4f}"
            )

    # --------------------------------------------------------
    # SAVE MODEL AND CHECKPOINT
    # --------------------------------------------------------

    if episode % SAVE_EVERY == 0:
        model.save_weights(MODEL_PATH)

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"actor_critic_ep_{episode}.weights.h5"
        )

        model.save_weights(checkpoint_path)

        print(
            f"\nSaved model and checkpoint at episode {episode}"
        )

    # --------------------------------------------------------
    # EVALUATE AGAINST FIXED OPPONENTS
    # --------------------------------------------------------

    if episode % EVALUATE_EVERY == 0:
        for opponent in (
            "advanced",
            "tactical",
            "random"
        ):
            evaluation = evaluate_against(
                model,
                opponent
            )

            print_evaluation(
                opponent,
                evaluation
            )


# ============================================================
# FINAL SAVE AND EVALUATION
# ============================================================

model.save_weights(MODEL_PATH)

print("\nTraining complete.")
print(f"Final weights saved to {MODEL_PATH}")

for opponent in (
    "advanced",
    "tactical",
    "random"
):
    evaluation = evaluate_against(
        model,
        opponent
    )

    print_evaluation(
        opponent,
        evaluation
    )
