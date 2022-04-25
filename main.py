import pygame
import sys
import time
import copy
from random import randrange

# board variables

# colors
BG_COLOR      = (255, 255, 255)
FG_COLOR      = (0,   0,   0  )
RED           = (255, 0,   0  )
BLUE          = (0  , 0,   255)
PLAYER_COLORS = [RED, BLUE]

# dimensions
WIDTH = 800
HEIGHT = 600
RADIUS = 20
GAP = 120
OFFSET_X = 50
OFFSET_Y = 50
RECT_WIDTH = 10

DOWN = 0
SIDE = 1
N, M = 3, 3
has_first_move = True
diffs = [1, -1]

def init():
    global diffs

    if not has_first_move:
        diffs.reverse()

def empty_board():
    board = ([[0 for _ in range(M)] for _ in range(N - 1)],
             [[0 for _ in range(M - 1)] for _ in range(N)])

    rectangles = ([[] for _ in range(N - 1)],
                  [[] for _ in range(N)])

    for i in range(0, N - 1):
            for j in range(0, M):
                pos = (OFFSET_X + GAP * j - RECT_WIDTH/2, OFFSET_Y + GAP * i + RADIUS)
                rectangles[DOWN][i].append([pygame.Rect(pos, (RECT_WIDTH, GAP - 2 * RADIUS)), BG_COLOR])

    for i in range(0, N):
        for j in range(0, M - 1):
            pos = (OFFSET_X + GAP * j + RADIUS, OFFSET_Y + GAP * i - RECT_WIDTH/2)
            rectangles[SIDE][i].append([pygame.Rect(pos, (GAP - 2*RADIUS, RECT_WIDTH)), BG_COLOR])

    return board, rectangles

def square_edges(board, i, j):
    return [board[DOWN][i][j], board[DOWN][i][j + 1], board[SIDE][i][j], board[SIDE][i + 1][j]]

def is_square(board, i, j):
    e = square_edges(board, i, j)
    return e[0] != 0 and e[1] != 0 and e[2] != 0 and e[3] != 0

def square_owner(board, i, j):
    return max(square_edges(board, i, j)) % 2

def score(board):
    res = 0
    for i in range(0, N - 1):
        for j in range(0, M - 1):
            if is_square(board, i, j):
                res += diffs[square_owner(board, i, j)]

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

def rand_move(board, _):
    while True:
        i, j = randrange(N), randrange(M)
        if i < N - 1 and board[DOWN][i][j] == 0:
            return DOWN, i, j

        if j < M - 1 and board[SIDE][i][j] == 0:
            return SIDE, i, j

def draw(_, rectangles, screen):
    for i in range(N):
        for j in range(M):
            pos = (OFFSET_X + GAP * j, OFFSET_Y + GAP * i)
            pygame.draw.circle(screen, FG_COLOR, pos, RADIUS)

    for i in range(0, N - 1):
        for j in range(0, M):
            rect, color = rectangles[DOWN][i][j]
            pygame.draw.rect(screen, color, rect)

    for i in range(0, N):
        for j in range(0, M - 1):
            rect, color = rectangles[SIDE][i][j]
            pygame.draw.rect(screen, color, rect)

    pygame.display.update()

def user_move(board, rectangles):
    while True:
        event = pygame.event.wait(1)
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()

            for i in range(0, N - 1):
                for j in range(0, M):
                    rect, _ = rectangles[DOWN][i][j]

                    if rect.collidepoint(pos) and board[DOWN][i][j] == 0:
                        return DOWN, i, j

            for i in range(0, N):
                for j in range(0, M - 1):
                    rect, _ = rectangles[SIDE][i][j]

                    if rect.collidepoint(pos) and board[SIDE][i][j] == 0:
                        return SIDE, i, j

def main():
    init()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dots & Boxes")
    screen.fill(BG_COLOR)
    pygame.display.update()

    wait_for_move = [rand_move, rand_move]
    if not has_first_move:
        wait_for_move.reverse()

    board, rectangles = empty_board()
    draw(board, rectangles, screen)

    previous_move = None
    move_number = 1
    while not game_ended(board):
        print("Drawing board...")

        player_idx = move_number % 2

        w, i, j = wait_for_move[player_idx](board, rectangles)
        board[w][i][j] = move_number
        rectangles[w][i][j][1] = PLAYER_COLORS[player_idx]
        print("Player {} has made move: {}".format(player_idx, (w,i,j)))
        print("Score:", score(board))
        print("")

        if previous_move is not None:
            pw, pi, pj = previous_move
            rectangles[pw][pi][pj][1] = FG_COLOR

        draw(board, rectangles, screen)
        time.sleep(1)

        move_number += 1
        previous_move = (w, i, j)
    
    print("Final score: {}".format(score(board)))

if __name__ == "__main__":
    main()
