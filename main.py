import pygame
import sys
import random
import time
import copy
from random import randrange

INF = sys.maxsize

# board variables

# colors
THEMES = [
    ((255, 255, 255),
     (0,   0,   0  ),
     (255, 0,   0  ),
     (0  , 0,   255)),
    ((0,   0,   0  ),
     (255, 255, 255),
     (255, 0,   0  ),
     (0  , 0,   255))
]
BG_COLOR, FG_COLOR, RED, BLUE = THEMES[1]
PLAYER_COLORS = [RED, BLUE]
PLAYER_NAMES  = ["RED", "BLUE"]

# dimensions
WIDTH = 800
HEIGHT = 600
RADIUS = 20
GAP = 120
OFFSET_X = 50
OFFSET_Y = 50
RECT_WIDTH = 10

DEFAULT_MAX_DEPTH = 3
DOWN = 0
SIDE = 1
N, M = 7, 7
gain_vals = [-1, 1]

def empty_board():
    board = ([[0 for _ in range(M)] for _ in range(N - 1)],
             [[0 for _ in range(M - 1)] for _ in range(N)])

    rectangles = ([[] for _ in range(N - 1)],
                  [[] for _ in range(N)])

    for i in range(N - 1):
            for j in range(M):
                pos = (OFFSET_X + GAP * j - RECT_WIDTH/2, OFFSET_Y + GAP * i + RADIUS)
                rectangles[DOWN][i].append([pygame.Rect(pos, (RECT_WIDTH, GAP - 2 * RADIUS)), BG_COLOR])

    for i in range(N):
        for j in range(M - 1):
            pos = (OFFSET_X + GAP * j + RADIUS, OFFSET_Y + GAP * i - RECT_WIDTH/2)
            rectangles[SIDE][i].append([pygame.Rect(pos, (GAP - 2*RADIUS, RECT_WIDTH)), BG_COLOR])

    return board, rectangles

def square_edges(board, i, j):
    return [board[DOWN][i][j], board[DOWN][i][j + 1], board[SIDE][i][j], board[SIDE][i + 1][j]]

def edge_sum(board, i, j):
    edges = square_edges(board, i, j)

    res = 0
    for e in edges:
        res += (e != 0)

    return res

def is_square(board, i, j):
    return edge_sum(board, i, j) == 4

def square_owner(board, i, j):
    return max(square_edges(board, i, j)) % 2

def remaining_moves(board):
    res = 0

    for i in range(N - 1):
        for j in range(M):
            res += (board[DOWN][i][j] == 0)

    for i in range(N):
        for j in range(M - 1):
            res += (board[SIDE][i][j] == 0)

    return res

def score(board):
    res = 0
    for i in range(N - 1):
        for j in range(M - 1):
            if is_square(board, i, j):
                res += gain_vals[square_owner(board, i, j)]

    return res

def heuristic_v1(board):
    return score(board)

def game_ended(board):
        for i in range(N - 1):
            for j in range(M):
                if board[DOWN][i][j] == 0:
                    return False

        for i in range(N):
            for j in range(M - 1):
                if board[SIDE][i][j] == 0:
                    return False

        return True

def draw(_, rectangles, screen):
    for i in range(N):
        for j in range(M):
            pos = (OFFSET_X + GAP * j, OFFSET_Y + GAP * i)
            pygame.draw.circle(screen, FG_COLOR, pos, RADIUS)

    for i in range(N - 1):
        for j in range(M):
            rect, color = rectangles[DOWN][i][j]
            pygame.draw.rect(screen, color, rect)

    for i in range(N):
        for j in range(M - 1):
            rect, color = rectangles[SIDE][i][j]
            pygame.draw.rect(screen, color, rect)

    pygame.display.update()

class ComputerPlayer:
    def __init__(self, method, heuristic=None, max_depth=None):
        self.method    = method
        self.heuristic = heuristic
        self.max_depth = max_depth

    def __call__(self, board):
        if self.heuristic is None:
            return self.method(board)
        if self.max_depth is None:
            return self.method(board, self.heuristic, DEFAULT_MAX_DEPTH)
        return self.method(board, self.heuristic, self.max_depth)

class Node:
    def __init__(self, board):
        self.board = board

    def neighbours(self, move_number):
        res = []

        # neighbours with new down edges
        for i in range(N - 1):
            for j in range(M):
                if self.board[DOWN][i][j] == 0:
                    new_down = copy.deepcopy(self.board[DOWN])
                    new_down[i][j] = move_number
                    res.append(((DOWN, i, j), Node((new_down, self.board[SIDE]))))

        # neighbours with new side edges
        for i in range(N):
            for j in range(M - 1):
                if self.board[SIDE][i][j] == 0:
                    new_side = copy.deepcopy(self.board[SIDE])
                    new_side[i][j] = move_number
                    res.append(((SIDE, i, j), Node((self.board[DOWN], new_side))))

        random.shuffle(res)

        return res

def rand_move(state):
    board, _, _ = state

    while True:
        i, j = randrange(N), randrange(M)
        if i < N - 1 and board[DOWN][i][j] == 0:
            return (DOWN, i, j), None

        if j < M - 1 and board[SIDE][i][j] == 0:
            return (SIDE, i, j), None

def user_move(state):
    board, rectangles, _ = state

    while True:
        event = pygame.event.wait(1)
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()

            for i in range(N - 1):
                for j in range(M):
                    rect, _ = rectangles[DOWN][i][j]

                    if rect.collidepoint(pos) and board[DOWN][i][j] == 0:
                        return (DOWN, i, j), None

            for i in range(N):
                for j in range(M - 1):
                    rect, _ = rectangles[SIDE][i][j]

                    if rect.collidepoint(pos) and board[SIDE][i][j] == 0:
                        return (SIDE, i, j), None

def minimax(state, heuristic, max_depth):
    _, _, move_number = state
    player_idx = move_number % 2

    if gain_vals[player_idx] > 0:
        return minimax_impl(state, max_depth, heuristic, True)
    else:
        return minimax_impl(state, max_depth, heuristic, False)


def minimax_impl(state, current_depth, heuristic, maximizing=True):
    board, _, move_number = state
    src = Node(board)

    if current_depth == 0:
        return None, heuristic(src.board)

    neighbours = src.neighbours(move_number)
    if len(neighbours) == 0:
        return None, heuristic(src.board)

    move, s = None, None
    if maximizing:
        max_val = -INF

        for via, v in neighbours:
            _, val = minimax_impl((v.board, None, move_number + 1),
                                  current_depth - 1,
                                  heuristic,
                                  False)
            if max_val < val:
                max_val = val
                move = via

        s = max_val
    else:
        min_val = INF

        for via, v in neighbours:
            _, val = minimax_impl((v.board, None, move_number + 1),
                                  current_depth - 1,
                                  heuristic,
                                  True)
            if min_val > val:
                min_val = val
                move = via

        s = min_val

    return move, s

def alpha_beta(state, heuristic, max_depth):
    _, _, move_number = state
    player_idx = move_number % 2

    if gain_vals[player_idx] > 0:
        return alpha_beta_impl(state, max_depth, -INF, INF, heuristic, True)
    else:
        return alpha_beta_impl(state, max_depth, -INF, INF, heuristic, False)

def alpha_beta_impl(state, current_depth, alpha, beta, heuristic, maximizing=True):
    board, _, move_number = state
    src = Node(board)

    if current_depth == 0:
        return None, heuristic(src.board)

    neighbours = src.neighbours(move_number)
    if len(neighbours) == 0:
        return None, heuristic(src.board)

    move, s = None, None
    if maximizing:
        max_val = -INF

        for via, v in neighbours:
            _, val = alpha_beta_impl((v.board, None, move_number + 1),
                                     current_depth - 1,
                                     alpha,
                                     beta,
                                     heuristic,
                                     False)
            if max_val < val:
                max_val = val
                move = via

            if max_val >= beta:
                break

            alpha = max(alpha, max_val)

        s = max_val
    else:
        min_val = INF

        for via, v in neighbours:
            _, val = alpha_beta_impl((v.board, None, move_number + 1),
                                     current_depth - 1,
                                     alpha,
                                     beta,
                                     heuristic,
                                     True)
            if min_val > val:
                min_val = val
                move = via

            if min_val <= alpha:
                break

            beta = min(beta, min_val)

        s = min_val

    return move, s

def lazy_alpha_beta(state, heuristic, max_depth):
    board, _, _ = state
    if remaining_moves(board) > 2 * max_depth:
        return rand_move(state)
    return alpha_beta(state, heuristic, max_depth)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dots & Boxes")
    screen.fill(BG_COLOR)
    pygame.display.update()

    wait_for_move = [
        ComputerPlayer(alpha_beta, heuristic_v1, 3),
        ComputerPlayer(alpha_beta, heuristic_v1, 3),
    ]

    board, rectangles = empty_board()
    draw(board, rectangles, screen)

    previous_move = None
    move_number = 1
    while not game_ended(board):
        player_idx = move_number % 2

        start_time = time.time()
        (w, i, j), _ = wait_for_move[player_idx]((board, rectangles, move_number))
        duration = time.time() - start_time

        board[w][i][j] = move_number
        rectangles[w][i][j][1] = PLAYER_COLORS[player_idx]

        print("After thinking time: {:.3f} seconds:".format(duration))
        print("Player {} has made move: {}".format(PLAYER_NAMES[player_idx], (w,i,j)))
        print("Score:", score(board))
        print("")

        if previous_move is not None:
            pw, pi, pj = previous_move
            rectangles[pw][pi][pj][1] = FG_COLOR

        draw(board, rectangles, screen)
        # time.sleep(1)

        move_number += 1
        previous_move = (w, i, j)
    
    fscore = score(board)
    print("Final score: {}".format(fscore))
    if fscore == 0:
        print("GAME ENDED IN A DRAW")
    else:
        print("{} WON!".format(PLAYER_NAMES[fscore > 0]))

    while True:
        event = pygame.event.wait(1)
        if event.type == pygame.QUIT:
            pygame.quit()
            return

if __name__ == "__main__":
    main()
