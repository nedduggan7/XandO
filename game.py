
import numpy as np


class Game:
    def __init__(self, size=3, win_count=3):
        if size < 1:
            raise ValueError("Board size must be at least 1.")

        if not 1 <= win_count <= size:
            raise ValueError(
                "Win count must be between 1 and board size."
            )

        self.size = size
        self.win_count = win_count
        self.reset()

    def reset(self):
        """Start a new game and return the empty board."""
        self.board = np.zeros(
            (self.size, self.size), dtype=int
        )
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()

    def get_legal_moves(self):
        """Return flat indices of all empty cells."""
        if self.done:
            return []

        return np.flatnonzero(
            self.board.reshape(-1) == 0
        ).tolist()

    def check_win(self, row, col, symbol):
        """Check whether the latest move creates a winning line."""
        m, n = self.board.shape

        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal down-right
            (1, -1)   # Diagonal down-left
        ]

        for dr, dc in directions:
            count = 1

            for sign in (1, -1):
                for step in range(1, self.win_count):
                    r = row + sign * dr * step
                    c = col + sign * dc * step

                    if not (0 <= r < m and 0 <= c < n):
                        break

                    if self.board[r, c] != symbol:
                        break

                    count += 1

            if count >= self.win_count:
                return True

        return False

    def step(self, move):
        """
        Apply a move for the current player.

        Parameters
        ----------
        move : int
            Flat board index, from 0 to size**2 - 1.

        Returns
        -------
        board : ndarray
            Updated board.
        reward : float
            +1 for a win, 0 for a draw or ongoing game.
        done : bool
            Whether the game has ended.
        info : dict
            Contains the winner when applicable.
        """
        if self.done:
            raise ValueError("The game has already ended.")

        legal_moves = self.get_legal_moves()

        if not isinstance(move, (int, np.integer)):
            raise ValueError("Move must be an integer index.")

        if move not in legal_moves:
            raise ValueError("Illegal move.")

        row, col = divmod(int(move), self.size)
        player = self.current_player

        self.board[row, col] = player

        # Check for a win before checking for a draw
        if self.check_win(row, col, player):
            self.done = True
            self.winner = player
            reward = 1.0

        elif not np.any(self.board == 0):
            self.done = True
            self.winner = None
            reward = 0.0

        else:
            reward = 0.0
            self.current_player *= -1

        return (
            self.board.copy(),
            reward,
            self.done,
            {"winner": self.winner}
        )
