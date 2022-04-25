import copy
from random import randrange

DOWN = 0
SIDE = 1
N, M = 3, 3
has_first_move = True
diffs = [-1, 1]

def init():
    global diffs

    if not has_first_move:
        diffs = [1, -1]

def empty_board():
    return ([[0 for _ in range(M)] for _ in range(N - 1)],
            [[0 for _ in range(M - 1)] for _ in range(N)])

def square_edges(board, i, j):
    return [board[DOWN][i][j], board[DOWN][i][j + 1], board[SIDE][i][j], board[SIDE][i + 1][j]]

def is_square(board, i, j):
    e = square_edges(board, i, j)
    return e[0] and e[1] and e[2] and e[3]

def square_owner(board, i, j):
    return max(square_edges(board, i, j)) % 2

def score(board):
    diff = [-1, 1]

    res = 0
    for i in range(0, N - 1):
        for j in range(0, M - 1):
            if is_square(board, i, j):
                res += diff[square_owner(board, i, j)]

    return res

def game_ended(board):
        for i in range(0, N - 1):
            for j in range(0, M):
                if board[DOWN][i][j] == 0:
                    return False

        for i in range(0, N):
            for j in range(0, M - 1):
                if board[SIDE][i][j] == 0:
                    return False

        return True


class Node:
    def __init__(self, board):
        self.board = board

    def neighbours(self, move_number):
        down_edges, side_edges = self.board[0], self.board[1]

        res = []

        # neighbours with new down edges
        for i in range(0, N - 1):
            for j in range(0, M):
                if down_edges[i][j] == 0:
                    new_down = copy.deepcopy(down_edges)
                    new_down[i][j] = move_number
                    res.append(Node((new_down, side_edges)))

        # neighbours with new side edges
        for i in range(0, N):
            for j in range(0, M - 1):
                if side_edges[i][j] == 0:
                    new_side = copy.deepcopy(side_edges)
                    new_side[i][j] = move_number
                    res.append(Node((down_edges, new_side)))

        return res

def rand_move(board):
    while True:
        i, j = randrange(N), randrange(M)
        if i < N - 1 and board[DOWN][i][j] == 0:
            return DOWN, i, j

        if j < M - 1 and board[SIDE][i][j] == 0:
            return SIDE, i, j

def main():
    init()

    ask_for_move = [rand_move, rand_move]

    board = empty_board()
    move = 1
    while not game_ended(board):
        print("Drawing board...")

        w, i, j = ask_for_move[move % 2](board)
        board[w][i][j] = move

        move += 1
    
    print(score(board))

if __name__ == "__main__":
    main()
