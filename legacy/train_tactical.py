
import os
import random

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

EPISODES = 10000
PRINT_EVERY = 100
EVALUATE_EVERY = 500
SAVE_EVERY = 500

LEARNING_RATE = 0.0003
GAMMA = 0.99

VALUE_COEF = 0.5
ENTROPY_COEF = 0.01

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

def get_tactical_target(game, player):
    """
    Return an exact tactical move when one is available.

    Priority:
    1. Take an immediate win.
    2. Block a single immediate opponent winning move.
    3. Otherwise, return None and let the CNN choose.

    If the opponent has multiple immediate winning moves, a single
    block cannot generally stop all of them, so no block is forced.
    """

    own_wins = find_winning_moves(game, player)

    if own_wins:
        # Any of these moves wins immediately.
        return own_wins[0]

    opponent_wins = find_winning_moves(
        game,
        -player
    )

    if len(opponent_wins) == 1:
        return opponent_wins[0]

    return None


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

    # Exact tactical decisions take priority over the neural network.
    tactical_target = get_tactical_target(game, player)
    if tactical_target is not None:
        return int(tactical_target), state, True

    logits, _ = model(
        state[np.newaxis, ...],
        training=False
    )

    scores = logits[0].numpy().reshape(-1)
    legal_moves = game.get_legal_moves()

    if not legal_moves:
        raise RuntimeError("No legal moves available.")

    legal_scores = scores[legal_moves]

    if not stochastic:
        index = int(np.argmax(legal_scores))
        return int(legal_moves[index]), state, False

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

    return move, state, False


# ============================================================
# BASIC TACTICAL OPPONENT
# ============================================================

def select_tactical_move(game):
    # Take an immediate win.
    winning_moves = find_winning_moves(
        game,
        game.current_player
    )

    if winning_moves:
        return winning_moves[0]

    # Block an immediate opponent win when possible.
    opponent_wins = find_winning_moves(
        game,
        -game.current_player
    )

    if len(opponent_wins) == 1:
        return opponent_wins[0]

    # Otherwise choose randomly.
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

            if not was_forced:
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
        turn_number
    )


# ============================================================
# ACTOR-CRITIC UPDATE
# ============================================================

def train_on_game(
    model,
    optimizer,
    trajectories,
    rewards,
    total_turns
):
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

    # A game containing only forced tactical moves has no CNN decisions
    # to learn from, so there is no gradient update for that game.
    if not states_list:
        return None

    states = np.stack(states_list).astype(np.float32)

    actions = np.asarray(
        actions_list,
        dtype=np.int32
    )

    returns = tf.convert_to_tensor(
        returns_list,
        dtype=tf.float32
    )

    batch_size = len(states)
    number_of_squares = BOARD_SIZE * BOARD_SIZE

    # Reconstruct legal-move masks from each recorded board.
    occupied = (
        states[:, :, :, 0] + states[:, :, :, 1]
    ).reshape(batch_size, -1) > 0.5

    legal_mask = tf.convert_to_tensor(~occupied)

    with tf.GradientTape() as tape:
        logits, values = model(
            states,
            training=True
        )

        logits = tf.reshape(
            logits,
            (batch_size, number_of_squares)
        )

        # Prevent the policy from assigning probability to occupied squares.
        logits = tf.where(
            legal_mask,
            logits,
            tf.cast(-1e9, logits.dtype)
        )

        log_probs = tf.nn.log_softmax(
            logits,
            axis=-1
        )

        probabilities = tf.nn.softmax(
            logits,
            axis=-1
        )

        selected_log_probs = tf.gather(
            log_probs,
            actions,
            axis=1,
            batch_dims=1
        )

        advantages = tf.stop_gradient(
            returns - values
        )

        actor_loss = -tf.reduce_mean(
            selected_log_probs * advantages
        )

        critic_loss = tf.reduce_mean(
            tf.square(returns - values)
        )

        entropy = -tf.reduce_mean(
            tf.reduce_sum(
                probabilities * log_probs,
                axis=-1
            )
        )

        total_loss = (
            actor_loss
            + VALUE_COEF * critic_loss
            - ENTROPY_COEF * entropy
        )

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
        "total_loss": float(total_loss.numpy()),
        "cnn_decisions": len(states_list)
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
print("Opponent distribution:")

for name, weight in zip(
    OPPONENT_TYPES,
    OPPONENT_WEIGHTS
):
    print(f"  {name}: {weight}%")

print(f"Advanced opponent depth: {ADVANCED_DEPTH}")

for episode in range(1, EPISODES + 1):

    opponent_type = random.choices(
        OPPONENT_TYPES,
        weights=OPPONENT_WEIGHTS,
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
        total_turns
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
        total_turns
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
