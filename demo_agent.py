import os
from homework import Board, write_Board_to

def render(board, step):
    print('-------------------STEP_{}-------------------'.format(step))
    print("{}'s turn".format(board.side))
    print(board.board)
    print('---------------------------------------------')
    os.system('clear')

board = Board(filename='default.txt')
done = False
counter = 0
while not done:
    render(board, counter)
    best_move = board.alpha_beta_search()
    board = board.post_move_Board(best_move)
    done = board.game_state != 'CONT'
    counter += 1
