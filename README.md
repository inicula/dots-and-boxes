## dots-and-boxes

```
$ python3 main.py --help

USAGE: python3 main.py [OPTIONS]

OPTIONS:
--non-interactive                                run in non-interactive mode (no pygame elements)
--wait-between-moves <seconds>                   wait a number of seconds between moves
--swap                                           swap the two players before starting the game
--difficulty <type>                              choose the game difficulty (maximum search depth)
--p1 <player-type> [<heuristic> <max-depth>]     create the first player with the given parameters
--p2 <player-type> [<heuristic> <max-depth>]     create the second player with the given parameters
--print-board                                    print board configuration to stdout after each move
--rows <number>                                  specify the number of rows on the board (>= 2)
--columns <number>                               specify the number of columns on the board (>= 2)
--help                                           print information about usage and options

PLAYER TYPES:
human                take input from the user
alphabeta            search with alpha beta pruning
alphabeta_sorted     sort the nodes according to the heuristic before alpha beta pruning
minimax              search with minimax

HEURISTICS:
v1     same as the current score on the board
v2     add the number of almost-complete squares to the player's score
v3     -inf/+inf if the player can't win with all of the remainig squares, otherwise same as v2

DIFFICULTIES:
easy       maximum depth is 2
medium     maximum depth is 3
hard       maximum depth is 5

Example #1 (start default game - human vs. alphabeta, heuristic 2, max depth 3):
python3 main.py

Example #2 (start human vs. human game):
python3 main.py --p1 human --p2 human

Example #3 (start human vs. alpha beta, heuristic 1, max depth 3):
python3 main.py --p1 human --p2 alphabeta v1 3 --wait-between-moves 1

Example #4 (alpha beta, heuristic 2, max depth 4 vs. minimax, heuristic 3, max depth 3):
python3 main.py --p1 alphabeta v2 4 --p2 minimax v3 3

Example #5 (human vs. alphabeta, alphabeta takes first move):
python3 main.py --swap

Example #6 (use PyPy when in non-interactive mode for faster execution)
pypy3 main.py --p1 alphabeta_sorted v3 5 --p2 alphabeta v3 5 --non-interactive
```
