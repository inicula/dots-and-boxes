import sys
import random
import time
from copy import deepcopy
from os import environ
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = '1'
import pygame

# colors
THEMES = [
    ((255, 255, 255),
     (0,   0,   0  ),
     (255, 0,   0  ),
     (0,   0,   255),
     (227, 227, 227)),
    ((0,   0,   0  ),
     (255, 255, 255),
     (255, 0,   0  ),
     (0,   0,   255),
     (64,  64,  64 ))
]
BG_COLOR, FG_COLOR, RED, BLUE, GRAY = THEMES[0]
PLAYER_COLORS = [RED, BLUE]
PLAYER_NAMES  = ["RED", "BLUE"]

# dimensions
WIDTH      = 800
HEIGHT     = 600
RADIUS     = 20
GAP        = 120
RECT_WIDTH = 15

# search parameters
DEFAULT_MAX_DEPTH = 3
GAIN_VALS         = [-1, 1]

# enums
DOWN = 0
SIDE = 1

# board config
N = 4
M = 5
OFFSET_X = (WIDTH  - (GAP * (M - 1))) / 2
OFFSET_Y = (HEIGHT - (GAP * (N - 1))) / 2

# misc
INF               = sys.maxsize
discovered_nodes  = 0
non_interactive   = False

def fprint(fmt, *args):
    print(fmt.format(*args))

def fprinterr(fmt, *args):
    print(fmt.format(*args), file=sys.stderr)

def print_help():
    fprint("Usage: python3 main.py [OPTIONS]\n")

    fprint("Options:")
    fprint("{:<48} {}", "--non-interactive", "run in non-interactive mode (no pygame elements)")
    fprint("{:<48} {}", "--wait-between-moves <seconds>", "wait a number of seconds between moves")
    fprint("{:<48} {}", "--swap", "swap the two players before starting the game")
    fprint("{:<48} {}", "--difficulty <type>", "choose the game difficulty (maximum search depth)")
    fprint("{:<48} {}", "--p1 <player-type> [<heuristic> <max-depth>]", "create the first player with the given parameters")
    fprint("{:<48} {}", "--p2 <player-type> [<heuristic> <max-depth>]", "create the second player with the given parameters")
    fprint("{:<48} {}", "--help", "print information about usage and options")

    fprint("\nPlayer types:")
    fprint("{:<20} {}", "human", "take input from the user")
    fprint("{:<20} {}", "alphabeta", "search with alpha beta pruning")
    fprint("{:<20} {}", "alphabeta_sorted", "sort the nodes according to the heuristic before alpha beta pruning")
    fprint("{:<20} {}", "minimax", "search with minimax")

    fprint("\nHeuristics:")
    fprint("{:<6} {}", "v1", "same as the current score on the board")
    fprint("{:<6} {}", "v2", "add the number of almost-complete squares to the player's score")
    fprint("{:<6} {}", "v3", "-inf/+inf if the player can't win with all of the remainig squares, otherwise same as v2")

    fprint("\nDifficulties:")
    fprint("{:<10} {}", "easy", "maximum depth is 2")
    fprint("{:<10} {}", "medium", "maximum depth is 3")
    fprint("{:<10} {}", "hard", "maximum depth is 5")

    fprint("\nExample #1 (start human vs. human game):")
    fprint("python3 main.py --p1 human --p2 human")

    fprint("\nExample #2 (start human vs. alpha beta, heuristic 1, max depth 3):")
    fprint("python3 main.py --p1 human --p2 alphabeta v1 3 --wait-between-moves 1")

    fprint("\nExample #3 (alpha beta, heuristic 2, max depth 4 vs. minimax, heuristic 3, max depth 3):")
    fprint("python3 main.py --p1 alphabeta v2 4 --p2 minimax v3 3")

def empty_board():
    board = ([[0 for _ in range(M)] for _ in range(N - 1)],
             [[0 for _ in range(M - 1)] for _ in range(N)])

    rectangles = ([[] for _ in range(N - 1)],
                  [[] for _ in range(N)])

    for i in range(N - 1):
        for j in range(M):
            pos = (OFFSET_X + GAP * j - RECT_WIDTH/2,
                   OFFSET_Y + GAP * i + RADIUS)

            rect = [pygame.Rect(pos, (RECT_WIDTH, GAP - 2 * RADIUS)), GRAY]
            rectangles[DOWN][i].append(rect)

    for i in range(N):
        for j in range(M - 1):
            pos = (OFFSET_X + GAP * j + RADIUS,
                   OFFSET_Y + GAP * i - RECT_WIDTH/2)

            rect = [pygame.Rect(pos, (GAP - 2 * RADIUS, RECT_WIDTH)), GRAY]
            rectangles[SIDE][i].append(rect)

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

def remaining_squares(board):
    res = 0
    for i in range(N - 1):
        for j in range(M - 1):
            res += (not is_square(board, i, j))

    return res

def score(board):
    res = 0
    for i in range(N - 1):
        for j in range(M - 1):
            if is_square(board, i, j):
                res += GAIN_VALS[square_owner(board, i, j)]

    return res

def heuristic_v1(state):
    return score(state[0])

def heuristic_v2(state):
    board, _, move_number = state
    player_idx = move_number % 2

    almost_complete = 0
    for i in range(N - 1):
        for j in range(M - 1):
            almost_complete += (edge_sum(board, i, j) == 3)

    partial_score = almost_complete * GAIN_VALS[player_idx]
    return partial_score + score(board)

def heuristic_v3(state):
    board, _, move_number = state
    player_idx = move_number % 2

    s = score(board)
    rem = remaining_squares(board)

    if GAIN_VALS[player_idx] > 0:
        if s + rem < 0:
            return -INF + 1
    else:
        if s - rem > 0:
            return INF - 1

    return heuristic_v2(state)

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

def make_x_figure(i, j, color):
    p1 = (OFFSET_X + GAP * j + RADIUS,
          OFFSET_Y + GAP * i + RADIUS)

    p2 = (OFFSET_X + GAP * (j + 1) - RADIUS,
          OFFSET_Y + GAP * (i + 1) - RADIUS)

    p3 = (OFFSET_X + GAP * j + RADIUS,
          OFFSET_Y + GAP * (i + 1) - RADIUS)

    p4 = (OFFSET_X + GAP * (j + 1) - RADIUS,
          OFFSET_Y + GAP * i + RADIUS)

    return [(p1, p2, p3, p4), color]

def make_triangle_figure(i, j, color):
    p1 = (OFFSET_X + GAP * j + RADIUS,
          OFFSET_Y + GAP * (i + 1) - RADIUS)

    p2 = (OFFSET_X + GAP * (j + 1) - RADIUS,
          OFFSET_Y + GAP * (i + 1) - RADIUS)

    p3 = (OFFSET_X + GAP * (j + 1) - RADIUS - (GAP - 2 * RADIUS) / 2,
          OFFSET_Y + GAP * i + RADIUS)

    return [(p1, p2, p3), color]

def made_square(board, via):
    w, i, j = via
    res = []

    if 0 <= i < N - 1 and 0 <= j < M - 1 and is_square(board, i, j):
        res.append((i, j))

    if w == DOWN:
        k, l = i, j - 1
        if 0 <= k < N - 1 and 0 <= l < M - 1 and is_square(board, k, l):
            res.append((k, l))
    else:
        k, l = i - 1, j
        if 0 <= k < N - 1 and 0 <= l < M - 1 and is_square(board, k, l):
            res.append((k, l))

    if len(res) > 0:
        return res

    return None

def draw(rectangles, figures, screen):
    if non_interactive:
        return

    screen.fill(BG_COLOR)

    # draw the circles
    for i in range(N):
        for j in range(M):
            pos = (OFFSET_X + GAP * j, OFFSET_Y + GAP * i)
            pygame.draw.circle(screen, FG_COLOR, pos, RADIUS)

    # draw the down-edges
    for i in range(N - 1):
        for j in range(M):
            rect, color = rectangles[DOWN][i][j]
            pygame.draw.rect(screen, color, rect)

    # draw the side-edges
    for i in range(N):
        for j in range(M - 1):
            rect, color = rectangles[SIDE][i][j]
            pygame.draw.rect(screen, color, rect)

    # draw the players' figures
    for points, color in figures:
        if len(points) == 4:
            pygame.draw.line(screen, color, points[0], points[1], 10)
            pygame.draw.line(screen, color, points[2], points[3], 10)
        else:
            pygame.draw.polygon(screen, color, points)

    pygame.display.update()

class Player:
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
    def __init__(self, board, current_move, has_scored=None):
        self.board = board
        self.current_move = current_move
        self.has_scored = has_scored

    def state(self):
        return (self.board, None, self.current_move)

    def neighbours(self):
        global discovered_nodes

        move_number = self.current_move
        res = []
        increments = [1, 2]
        # neighbours with new down/side edges
        for way in [DOWN, SIDE]:
            for i in range(N - (not way)):
                for j in range(M - way):
                    if self.board[way][i][j] != 0:
                        continue

                    new_board = None
                    if way == SIDE:
                        new_board = (self.board[DOWN], deepcopy(self.board[SIDE]))
                    else:
                        new_board = (deepcopy(self.board[DOWN]), self.board[SIDE])

                    move = (way, i, j)
                    new_board[way][i][j] = move_number
                    made_sq = (made_square(new_board, move) is not None)

                    res.append((
                        move,
                        Node(new_board, move_number + increments[made_sq], made_sq)
                    ))

        # avoid deterministic Computer vs. Computer matches
        random.shuffle(res)

        discovered_nodes += len(res)
        return res

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
                    move = (DOWN, i, j)
                    rect, _ = rectangles[DOWN][i][j]

                    if rect.collidepoint(pos) and board[DOWN][i][j] == 0:
                        return move, None

            for i in range(N):
                for j in range(M - 1):
                    move = (SIDE, i, j)
                    rect, _ = rectangles[SIDE][i][j]

                    if rect.collidepoint(pos) and board[SIDE][i][j] == 0:
                        return move, None

def is_human(player):
    return player.method == user_move

def minimax(state, heuristic, max_depth):
    _, _, move_number = state
    idx = move_number % 2

    return minimax_impl(state, max_depth, heuristic, GAIN_VALS[idx] > 0)

def minimax_impl(state, current_depth, heuristic, maximizing=True):
    board, _, move_number = state
    src = Node(board, move_number)

    if current_depth == 0:
        return None, heuristic(state)

    neighbours = src.neighbours()
    if len(neighbours) == 0:
        return None, heuristic(state)

    move, s = None, None
    if maximizing:
        max_val = -INF

        for via, v in neighbours:
            _, val = minimax_impl(v.state(),
                                  current_depth - 1,
                                  heuristic,
                                  v.has_scored)

            if max_val < val:
                max_val = val
                move = via

        s = max_val
    else:
        min_val = INF

        for via, v in neighbours:
            _, val = minimax_impl(v.state(),
                                  current_depth - 1,
                                  heuristic,
                                  not v.has_scored)

            if min_val > val:
                min_val = val
                move = via

        s = min_val

    return move, s

def alpha_beta(state, heuristic, max_depth):
    _, _, move_number = state
    idx = move_number % 2

    return alpha_beta_impl(state, max_depth, -INF, INF, heuristic, GAIN_VALS[idx] > 0, False)

def alpha_beta_sorted(state, heuristic, max_depth):
    _, _, move_number = state
    idx = move_number % 2

    return alpha_beta_impl(state, max_depth, -INF, INF, heuristic, GAIN_VALS[idx] > 0, True)

def alpha_beta_impl(state, current_depth, alpha, beta, heuristic, maximizing=True, sort=False):
    board, _, move_number = state
    src = Node(board, move_number)

    if current_depth == 0:
        return None, heuristic(state)

    neighbours = src.neighbours()
    if len(neighbours) == 0:
        return None, heuristic(state)

    if sort and maximizing:
        neighbours = sorted(neighbours,
                            reverse=True,
                            key=lambda node:
                                    heuristic((node[1].board, None, node[1].current_move)))
    elif sort:
        neighbours = sorted(neighbours,
                            key=lambda node:
                                    heuristic((node[1].board, None, node[1].current_move)))

    move, s = None, None
    if maximizing:
        max_val = -INF
        for via, v in neighbours:
            _, val = alpha_beta_impl(v.state(),
                                     current_depth - 1,
                                     alpha,
                                     beta,
                                     heuristic,
                                     v.has_scored,
                                     sort)

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
            _, val = alpha_beta_impl(v.state(),
                                     current_depth - 1,
                                     alpha,
                                     beta,
                                     heuristic,
                                     not v.has_scored,
                                     sort)

            if min_val > val:
                min_val = val
                move = via
            if min_val <= alpha:
                break
            beta = min(beta, min_val)

        s = min_val

    return move, s

def main(argv):
    global discovered_nodes
    global non_interactive

    # tables for move methods and heuristics
    search_methods = {
        "alphabeta"        : alpha_beta,
        "alphabeta_sorted" : alpha_beta_sorted,
        "minimax"          : minimax,
        "human"            : user_move
    }

    heuristics = {
        "v1" : heuristic_v1,
        "v2" : heuristic_v2,
        "v3" : heuristic_v3,
    }

    difficulty_depth = {
        "easy"   : 2,
        "medium" : 3,
        "hard"   : 5
    }

    # default players
    players = [
        Player(alpha_beta, heuristic_v3, 3),
        Player(user_move)
    ]

    # parse cli args
    argc = len(argv)
    if argc >= 2 and argv[1] == "--help":
        print_help()
        return

    wait_dur = None
    swap_players = False
    try:
        manual_player_setting = False
        difficulty_setting = False

        i = 1
        while i < argc:
            if argv[i] == "--non-interactive":
                non_interactive = True

            if argv[i] == "--wait-between-moves":
                wait_dur = float(argv[i + 1])
                i += 1

            if argv[i] == "--swap":
                swap_players = True

            if argv[i] == "--difficulty":
                difficulty_setting = True
                players[0].max_depth = difficulty_depth[argv[i + 1]]
                i += 1

            if argv[i] == "--p1" or argv[i] == "--p2":
                manual_player_setting = True

                idx = int(argv[i][3]) - 1
                if not (0 <= idx <= 1):
                    raise Exception('')

                idx = 1 - idx
                if argv[i + 1] == "human":
                    players[idx] = Player(user_move)
                    i += 1
                else:
                    players[idx] = Player(search_methods[argv[i + 1]],
                                          heuristics[argv[i + 2]],
                                          int(argv[i + 3]))
                    i += 3

            i += 1

        if manual_player_setting and difficulty_setting:
            raise Exception('')
    except:
        fprinterr("Erorr in cli args.")
        fprinterr("Run with flag '--help' for usage details.")
        exit(1)

    if swap_players:
        players.reverse()

    # inits
    screen = None
    if not non_interactive:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Dots & Boxes")

    # table for making player figures
    make_player_figure = [
        make_triangle_figure,
        make_x_figure
    ]

    # no user_move when non-interactive
    if non_interactive:
        m1 = players[0].method
        m2 = players[1].method
        if m1 == user_move or m2 == user_move:
            fprinterr("error: user_move not defined in non-interactive mode")
            exit(1)

    # make empty board with free rectangles
    board, rectangles = empty_board()
    figures = []

    # draw twice (missing desktop environment?)
    draw(rectangles, figures, screen)
    draw(rectangles, figures, screen)

    # start the main loop
    previous_figure_idx = None
    previous_move = None
    move_number = 1
    while not game_ended(board):
        player_idx = move_number % 2

        # clean up from previous move
        discovered_nodes = 0

        # wait for the player's next move
        start_time = time.time()
        (w, i, j), _ = players[player_idx]((board, rectangles, move_number))
        duration = time.time() - start_time

        # put the move on the board
        board[w][i][j] = move_number
        rectangles[w][i][j][1] = PLAYER_COLORS[player_idx]

        # check if the new move created squares
        sq = made_square(board, (w, i, j))
        new_figures_idx = []
        make_figure = make_player_figure[player_idx]
        if sq is not None:
            for k, l in sq:
                figures.append(make_figure(k, l, PLAYER_COLORS[player_idx]))
                new_figures_idx.append(len(figures) - 1)

        # Print move information
        fprint("MOVE #{}:", move_number)
        fprint("Thinking time: {:.3f} seconds:", duration)
        if not is_human(players[player_idx]):
            fprint("Discovered nodes: {}", 1 + discovered_nodes)
        fprint("Player {} has made move: {}", PLAYER_NAMES[player_idx], (w,i,j))
        fprint("Score: {}", score(board))
        if players[player_idx].heuristic is not None:
            fprint("Estimated score: {}",
                   players[player_idx].heuristic((board, rectangles, move_number)))
        fprint("")

        # Don't highlight move from previous turn
        if previous_move is not None:
            pw, pi, pj = previous_move
            rectangles[pw][pi][pj][1] = FG_COLOR
        if previous_figure_idx is not None:
            for idx in previous_figure_idx:
                figures[idx][1] = FG_COLOR

        # Draw the board with the new move
        draw(rectangles, figures, screen)

        # Prepare for next iteration
        if sq is None:
            move_number += 1
        else:
            fprint("MOVE #{}: skipped\n", move_number + 1)
            move_number += 2
        previous_move = (w, i, j)
        previous_figure_idx = new_figures_idx

        if (wait_dur is not None) and (not is_human(players[move_number % 2])):
            time.sleep(wait_dur)
    
    # Print info at the end of the game
    fscore = score(board)
    fprint("Final score: {}", fscore)
    if fscore == 0:
        fprint("GAME ENDED IN A DRAW")
    else:
        fprint("{} WON!", PLAYER_NAMES[fscore > 0])

    # Wait for manual user exit
    while not non_interactive:
        event = pygame.event.wait(1)
        if event.type == pygame.QUIT:
            pygame.quit()
            return

if __name__ == "__main__":
    main(sys.argv)
