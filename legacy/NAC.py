
import numpy as np


def check_win(board, win_count, row, col, sym):
    m, n = board.shape

    directions = [
        (0, 1),   # Horizontal
        (1, 0),   # Vertical
        (1, 1),   # Diagonal down-right
        (1, -1)   # Diagonal down-left
    ]

    for dr, dc in directions:
        count = 1

        for sign in (1, -1):
            for step in range(1, win_count):
                r = row + sign * dr * step
                c = col + sign * dc * step

                if not (0 <= r < m and 0 <= c < n):
                    break

                if board[r, c] != sym:
                    break

                count += 1

        if count >= win_count:
            return True

    return False


# Game setup
while True:
    try:
        size = int(input("Select board size: "))
        win_count = int(input("Select win count: "))

        if size < 1 or win_count < 1:
            print("Both values must be positive.")
        elif win_count > size:
            print("Win count cannot exceed board size.")
        else:
            break

    except ValueError:
        print("Please enter integers.")

board = np.zeros((size, size), dtype=int)
turn = 0

print(board)

# Game loop
while True:
    try:
        pos = int(input(f"Select board position (1-{size**2}): "))
    except ValueError:
        print("Invalid input: enter an integer.")
        continue

    if not 1 <= pos <= size**2:
        print("Invalid position.")
        continue

    row, col = divmod(pos - 1, size)

    if board[row, col] != 0:
        print("Position already occupied.")
        continue

    # Player 1 uses 1; Player 2 uses 2
    player = 1 if turn % 2 == 0 else 2
    board[row, col] = player
    turn += 1

    print(board)

    if check_win(board, win_count, row, col, player):
        print(f"Player {player} wins!")
        break

    if turn == size**2:
        print("DRAW")
        break

print("Game over")
