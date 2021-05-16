import numpy as np
from homework import *

def one_step_to_seq(steps):


def read_move_as_index(movefile):
    with open(movefile, 'r') as f:
        while True:
            one = f.readline().strip()
            if line == '':
                # either end of file or just a blank line.....
                # we'll assume EOF, because we don't have a choice with the while loop!
                break

def read_and_move(boardfile='midgame.txt', movefile='output.txt'):
    currentBoard = Board(filename='midgame.txt')

    currentBoard.move

if __name__ == "__main__":
    runCase = Board()
    runCase.write_best_to()
