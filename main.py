# README.md: https://github.com/niculaionut/dots-and-boxes

import sys
import statistics
import random
import time
from copy import deepcopy
from os import environ
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = '1'

# Colors
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

# Dimensions
WIDTH      = 960
HEIGHT     = 1080
RADIUS     = 20
GAP        = 120
RECT_WIDTH = 15

# Search parameters
DEFAULT_MAX_DEPTH = 3
GAIN_VALS         = [-1, 1]

# Enums
DOWN = 0
SIDE = 1

# Board config
N = 5
M = 6
OFFSET_X = (WIDTH  - (GAP * (M - 1))) / 2
OFFSET_Y = (HEIGHT - (GAP * (N - 1))) / 2

# Misc
INF               = float('inf')
BIGVAL            = sys.maxsize / 4
discovered_nodes  = 0
non_interactive   = False
stats             = []
made_n_moves      = [0, 0]
g_start_time      = time.time()
g_end_time        = time.time()

def fprint(fmt, *args):
    print(fmt.format(*args))

def fprinterr(fmt, *args):
    print(fmt.format(*args), file=sys.stderr)

def print_help():
    fprint("USAGE: python3 main.py [OPTIONS]")
    fprint("       pypy3 main.py --non-interactive [OPTIONS]\n")

    fprint("OPTIONS:")
    fprint("{:<48} {}", "--non-interactive", "run in non-interactive mode (no pygame elements)")
    fprint("{:<48} {}", "--wait-between-moves <seconds>", "wait a number of seconds between moves")
    fprint("{:<48} {}", "--swap", "swap the two players before starting the game")
    fprint("{:<48} {}", "--difficulty <type>", "choose the game difficulty (maximum search depth)")
    fprint("{:<48} {}", "--p1 <player-type> [<heuristic> <max-depth>]", "create the first player with the given parameters")
    fprint("{:<48} {}", "--p2 <player-type> [<heuristic> <max-depth>]", "create the second player with the given parameters")
    fprint("{:<48} {}", "--print-board", "print board configuration to stdout after each move")
    fprint("{:<48} {}", "--rows <number>", "specify the number of rows on the board (>= 2)")
    fprint("{:<48} {}", "--columns <number>", "specify the number of columns on the board (>= 2)")
    fprint("{:<48} {}", "--help", "print information about usage and options")

    fprint("\nPLAYER TYPES:")
    fprint("{:<20} {}", "human", "take input from the user")
    fprint("{:<20} {}", "alphabeta", "search with alpha beta pruning")
    fprint("{:<20} {}", "alphabeta_sorted", "sort the nodes according to the heuristic before alpha beta pruning")
    fprint("{:<20} {}", "minimax", "search with minimax")

    fprint("\nHEURISTICS:")
    fprint("{:<6} {}", "v1", "same as the current score on the board")
    fprint("{:<6} {}", "v2", "add the number of almost-complete squares to the player's score")
    fprint("{:<6} {}", "v3", "-inf/+inf if the player can't win with all of the remainig squares, otherwise same as v2")

    fprint("\nDIFFICULTIES:")
    fprint("{:<10} {}", "easy", "maximum depth is 2")
    fprint("{:<10} {}", "medium", "maximum depth is 3")
    fprint("{:<10} {}", "hard", "maximum depth is 5")

    fprint("\nExample #1 (start default game - human vs. alphabeta, heuristic 2, max depth 3):")
    fprint("python3 main.py")

    fprint("\nExample #2 (start human vs. human game):")
    fprint("python3 main.py --p1 human --p2 human")

    fprint("\nExample #3 (start human vs. alpha beta, heuristic 1, max depth 3):")
    fprint("python3 main.py --p1 human --p2 alphabeta v1 3 --wait-between-moves 1")

    fprint("\nExample #4 (alpha beta, heuristic 2, max depth 4 vs. minimax, heuristic 3, max depth 3):")
    fprint("python3 main.py --p1 alphabeta v2 4 --p2 minimax v3 3")

    fprint("\nExample #5 (human vs. alphabeta, alphabeta takes first move):")
    fprint("python3 main.py --swap")

    fprint("\nExample #6 (use PyPy when in non-interactive mode for faster execution):")
    fprint("pypy3 main.py --p1 alphabeta_sorted v3 5 --p2 alphabeta v3 5 --non-interactive")

def empty_board():
    # Generate the initial board/game state

    board = ([[0 for _ in range(M)] for _ in range(N - 1)],
             [[0 for _ in range(M - 1)] for _ in range(N)])

    if non_interactive:
        return (board, [])
    else:
        import pygame

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
    # Get segments that form square at position (i, j) (starting with the upper-left point)

    return [board[DOWN][i][j], board[DOWN][i][j + 1], board[SIDE][i][j], board[SIDE][i + 1][j]]

def edge_sum(board, i, j):
    edges = square_edges(board, i, j)

    res = 0
    for e in edges:
        res += (e != 0)

    return res

def is_square(board, i, j):
    # Check if a complete square starts at position (i, j) (upper left corner)

    return edge_sum(board, i, j) == 4

def square_owner(board, i, j):
    # The owner of a complete square is the player that made the most recent move
    # accross one of the square's segments

    return max(square_edges(board, i, j)) % 2

def remaining_squares(board):
    # Get number of remaining squares on the board

    res = 0
    for i in range(N - 1):
        for j in range(M - 1):
            res += (not is_square(board, i, j))

    return res

def score(board):
    # Calculate score
    # Each owned square has value 1
    # For Player 1, the value is added to the score
    # For Player 2, the value is subtracted from the score

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
    board, _, _ = state

    s = score(board)
    rem = remaining_squares(board)

    if s + rem < 0:
        return -BIGVAL
    if s - rem > 0:
        return BIGVAL

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
    # Check if move `via` created squares on the board

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
    else:
        import pygame

    screen.fill(BG_COLOR)

    # Draw the circles
    for i in range(N):
        for j in range(M):
            pos = (OFFSET_X + GAP * j, OFFSET_Y + GAP * i)
            pygame.draw.circle(screen, FG_COLOR, pos, RADIUS)

    # Draw the down-edges
    for i in range(N - 1):
        for j in range(M):
            rect, color = rectangles[DOWN][i][j]
            pygame.draw.rect(screen, color, rect)

    # Draw the side-edges
    for i in range(N):
        for j in range(M - 1):
            rect, color = rectangles[SIDE][i][j]
            pygame.draw.rect(screen, color, rect)

    # Draw the players' figures
    for points, color in figures:
        if len(points) == 4:
            pygame.draw.line(screen, color, points[0], points[1], 10)
            pygame.draw.line(screen, color, points[2], points[3], 10)
        else:
            pygame.draw.polygon(screen, color, points)

    pygame.display.update()

def board_to_str(board):
    # Get string representation of the board

    fmt = "Board configuration:\nDown edges matrix:\n{}\n\nSide edges matrix:\n{}"
    return fmt.format("\n".join(map(str, board[DOWN])),
                      "\n".join(map(str, board[SIDE])))

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
        # Generate node neighbours

        global discovered_nodes

        move_number = self.current_move
        res = []
        increments = [1, 2]

        # Neighbours with new down/side edges
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

        # Avoid deterministic Computer vs. Computer matches
        random.shuffle(res)

        discovered_nodes += len(res)
        return res

class Game_stats:
    def __init__(self, player):
        self.player = player
        self.thinking_time = []
        self.discovered = []

    def print(self):
        thinking_time = self.thinking_time
        discovered = self.discovered

        thinking_time = sorted(thinking_time)
        if len(thinking_time) > 0:
            fprint("Thinking time (seconds):")
            fprint("min: {:.3f}\nmax: {:.3f}\naverage: {:.3f}\nmedian: {:.3f}",
                  min(thinking_time),
                  max(thinking_time),
                  statistics.mean(thinking_time),
                  statistics.median(thinking_time))

        if is_human(self.player):
            return

        if len(discovered) > 0:
            print("\nDiscovered nodes:")
            fprint("min: {}\nmax: {}\naverage: {:.1f}\nmedian: {:.1f}",
                  min(discovered),
                  max(discovered),
                  statistics.mean(discovered),
                  statistics.median(discovered))

def print_end_info():
    # Print information at the end of the game

    fprint("GAME ENDED!\nDuration: {:.2f} seconds", g_end_time - g_start_time)
    fprint("Player 1 made {} moves", made_n_moves[1])
    fprint("Player 2 made {} moves", made_n_moves[0])

    fprint("\nPlayer 1:")
    stats[1].print()

    fprint("\nPlayer 2:")
    stats[0].print()

def user_move(state):
    # Get the user's next move

    import pygame
    global g_end_time

    board, rectangles, _ = state
    while True:
        event = pygame.event.wait(1)
        if event.type == pygame.QUIT:
            g_end_time = time.time()
            print_end_info()
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
    global stats
    global made_n_moves
    global g_start_time
    global g_end_time
    global N
    global M

    # Tables for move methods and heuristics
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

    # Default players
    players = [
        Player(alpha_beta, heuristic_v3, 3),
        Player(user_move)
    ]

    # Parse cli args
    argc = len(argv)
    if argc >= 2 and argv[1] == "--help":
        print_help()
        return

    wait_dur = None
    swap_players = False
    print_board = False
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

            if argv[i] == "--print-board":
                print_board = True

            if argv[i] == "--rows":
                N = int(argv[i + 1])
                i += 1

            if argv[i] == "--columns":
                M = int(argv[i + 1])
                i += 1

            i += 1

        if manual_player_setting and difficulty_setting:
            raise Exception('')

        if not (N >= 2 and M >= 2):
            raise Exception('')

    except:
        fprinterr("Erorr in cli args.")
        fprinterr("Run with flag '--help' for usage details.")
        exit(1)

    if swap_players:
        players.reverse()

    # Inits
    screen = None
    if not non_interactive:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Nicula Ionut / Dots & Boxes")

    # Table for making player figures
    make_player_figure = [
        make_triangle_figure,
        make_x_figure
    ]

    # No user_move when non-interactive
    if non_interactive:
        m1 = players[0].method
        m2 = players[1].method
        if m1 == user_move or m2 == user_move:
            fprinterr("error: user_move not defined in non-interactive mode")
            exit(1)

    # Make empty board with free rectangles
    #
    # If `board[DOWN][i][j]` is non-zero, the position (i, j) is connected
    # to (i + 1, j)
    #
    # If `board[SIDE][i][j]` is non-zero, the position (i, j) is connected
    # to (i, j + 1)
    #
    # Elements in the `board` matrix are move numbers (so the program can
    # keep track of which player made which move)
    #
    # The first player always has move numbers that are odd, and the second
    # player always has move numbers that are even
    board, rectangles = empty_board()
    figures = []

    # Draw twice (missing desktop environment?)
    draw(rectangles, figures, screen)
    draw(rectangles, figures, screen)

    # Table for game statistics
    stats = [
        Game_stats(players[0]),
        Game_stats(players[1])
    ]

    # Start the main loop
    previous_figure_idx = None
    previous_move = None
    move_number = 1
    g_start_time = time.time()
    while True:
        player_idx = move_number % 2
        made_n_moves[player_idx] += 1

        # Clean up from previous move
        discovered_nodes = 0

        # Wait for the player's next move
        start_time = time.time()
        (w, i, j), _ = players[player_idx]((board, rectangles, move_number))
        duration = time.time() - start_time

        # Add to game stats
        stats[player_idx].thinking_time.append(duration)
        stats[player_idx].discovered.append(discovered_nodes)

        # Put the move on the board
        board[w][i][j] = move_number
        if not non_interactive:
            rectangles[w][i][j][1] = PLAYER_COLORS[player_idx]

        # Check if the new move created squares
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
            estimated = players[player_idx].heuristic((board, None, move_number))
            if estimated == BIGVAL:
                estimated = float('inf')
            elif estimated == -BIGVAL:
                estimated = float('-inf')
            fprint("Estimated score: {}", estimated)

        if print_board:
            fprint("{}", board_to_str(board))
        fprint("")

        # Don't highlight move from previous turn
        if previous_move is not None:
            pw, pi, pj = previous_move
            if not non_interactive:
                rectangles[pw][pi][pj][1] = FG_COLOR
        if previous_figure_idx is not None:
            for idx in previous_figure_idx:
                figures[idx][1] = FG_COLOR

        # Draw the board with the new move
        draw(rectangles, figures, screen)

        # Check if game has ended
        if game_ended(board):
            g_end_time = time.time()
            break

        # Prepare for next iteration
        # Skip opponent move if the most recent segment scored any points
        if sq is None:
            move_number += 1
        else:
            fprint("MOVE #{}: skipped\n", move_number + 1)
            move_number += 2

        previous_move = (w, i, j)
        previous_figure_idx = new_figures_idx

        if (wait_dur is not None) and (not is_human(players[move_number % 2])):
            time.sleep(wait_dur)

    # Print game stats
    print_end_info()

    # Print info at the end of the game
    fscore = score(board)
    fprint("\nFinal score: {}", fscore)
    if fscore == 0:
        fprint("GAME ENDED IN A DRAW")
    else:
        fprint("{} WON!", PLAYER_NAMES[fscore > 0])

    # Wait for manual user exit
    if non_interactive:
        return

    import pygame
    while True:
        event = pygame.event.wait(1)
        if event.type == pygame.QUIT:
            pygame.quit()
            return

if __name__ == "__main__":
    main(sys.argv)
