# XandObot

XandObot is a two-player, free-placement four-in-a-row game on a **6 × 6** board. Players take turns placing a piece in any empty square; the first to make four consecutive pieces horizontally, vertically, or diagonally wins. This is not gravity-based Connect Four.

The project includes a TensorFlow actor–critic model, a Tkinter interface for playing against it, online learning from human games, and a mixed-opponent training program.

## Quick start

Run commands from the project directory so model and checkpoint paths resolve correctly.

### Install dependencies

The main third-party dependencies are TensorFlow and NumPy. Tkinter is part of Python’s standard library on most Windows installations.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorflow numpy
```

Use a Python release supported by the TensorFlow package available for your operating system. If Tkinter is missing, install the Python distribution’s Tk support.

### Play the model

```powershell
python play_gui.py
```

Choose whether you or the model starts, then click **Start Game**. The GUI uses the saved `actor_critic.weights.h5` weights and applies the same tactical move rules as the trainer. Online learning starts **OFF**. Turn it on with the button if you want completed games to update and save the model weights.

### Train the model

```powershell
python train.py
```

Training starts immediately and runs 1,000 episodes. If `actor_critic.weights.h5` exists, training loads those weights; otherwise it starts with random weights. It saves the main weights and a checkpoint every 500 episodes in `checkpoints/`. This loads the weights only: optimizer state and episode count start fresh on each run.

Training opponents are:

- Advanced minimax opponent, depth 4
- Tactical opponent
- Random opponent
- Self-play

The advanced opponent’s share ramps from 10% to 30% during training. The final mix is 30% advanced, 50% tactical, 10% random, and 10% self-play. Training also evaluates against the advanced, tactical, and random opponents every 500 episodes.

### Evaluate saved weights

```powershell
python evaluate_current.py
```

This plays 10 games against each opponent (five with the model starting first and five second), including self-play. The advanced opponent uses depth 4. Results are written to `evaluation_current_results.csv`, replacing that file on each run. Evaluation uses the model together with its hard-coded tactical rules; it does not measure the CNN in isolation.

## Tactical move rules

The game-playing policy applies these priorities before considering unrestricted moves:

1. Take an immediate win.
2. Block an opponent’s immediate win.
3. If possible, create an open three: three consecutive pieces with an empty square at both ends.
4. Otherwise, prevent the opponent from setting up an open three.

When multiple squares satisfy the active rule, the model’s policy scores rank those squares. If no tactical rule applies, it selects among legal empty squares.

## Project files

| File or folder | Purpose |
| --- | --- |
| `play_gui.py` | Tkinter game interface with optional online learning. |
| `train.py` | Actor–critic training on the mixed opponent set. |
| `evaluate_current.py` | Evaluation of saved weights on the 6 × 6 game. |
| `game.py` | Shared board state, legal moves, win detection, and turn handling. |
| `game_model.py` | CNN actor–critic architecture: per-square policy logits and a board-value estimate. |
| `advanced_opponent.py` | Depth-limited minimax opponent with alpha-beta pruning and a transposition table. |
| `actor_critic.weights.h5` | Main saved model weights used by the GUI, trainer, and evaluator. |
| `checkpoints/` | Periodic model-weight snapshots created by training. |
| `evaluation_current_results.csv` | Per-game output from the most recent evaluation. |
| `legacy/` | Earlier scripts and alternate model files retained for reference; not used by the active scripts. |

## Git notes

The GUI and evaluator need `actor_critic.weights.h5` to load a trained model. Checkpoints are optional snapshots and can make a repository grow quickly. The local `.venv/`, `__pycache__/`, and IDE settings are machine-specific and generally should not be committed. Review `legacy/` and evaluation output before publishing if you want a smaller repository.
