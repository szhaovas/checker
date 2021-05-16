import copy
import random
import numpy as np

basic_board = np.array(
    [['.','b','.','b','.','b','.','b'],
     ['b','.','b','.','b','.','b','.'],
     ['.','b','.','b','.','b','.','b'],
     ['.','.','.','.','.','.','.','.'],
     ['.','.','.','.','.','.','.','.'],
     ['w','.','w','.','w','.','w','.'],
     ['.','w','.','w','.','w','.','w'],
     ['w','.','w','.','w','.','w','.']])

output_indices = np.array([[chr(x)+str(i) for x in range(ord('a'), ord('h')+1)] for i in range(8,0,-1)])

def move(board, fr, to):
    piece = board[fr]
    board[fr] = '.'
    mid_y = (to[0]+fr[0])/2.0
    # jump move
    if mid_y.is_integer():
        mid_x = (to[1]+fr[1])/2.0
        mid = (int(mid_y), int(mid_x))
        mid_piece = board[mid]
        board[mid] = '.'
    # simple move
    board[to] = piece
    # no need to crown king because this is used only for possible_jumps and kings cannot move backwards THIS turn

def valid_simple_move(board, fr, to):
    fr_y = fr[0]
    fr_x = fr[1]
    to_y = to[0]
    to_x = to[1]

    # is a simple move
    fr_piece = board[fr]
    if abs(to_y-fr_y) != 1 or abs(to_x-fr_x) != 1 or fr_piece == '.':
        return False
    # within bounds
    if to_y > 7 or to_x > 7 or to_y < 0 or to_x < 0:
        return False
    # target square blank
    if board[to] != '.':
        return False

    # white king or black king
    if fr_piece == 'W' or fr_piece == 'B':
        return True
    # white pawn
    elif fr_piece == 'w':
        # white pawns only move up
        return True if (to_y - fr_y == -1) else False
    # black pawn
    elif fr_piece == 'b':
        # black pawns only move down
        return True if (to_y - fr_y == 1) else False

def valid_jump(board, fr, to):
    fr_y = fr[0]
    fr_x = fr[1]
    to_y = to[0]
    to_x = to[1]

    # is a jump
    fr_piece = board[fr]
    if abs(to_y-fr_y) != 2 or abs(to_x-fr_x) != 2 or fr_piece == '.':
        return False
    # within bounds
    if to_y > 7 or to_x > 7 or to_y < 0 or to_x < 0:
        return False
    # target square blank
    if board[to] != '.':
        return False

    mid_piece = board[int((to_y+fr_y)/2.0), int((to_x+fr_x)/2.0)]
    # white king
    if fr_piece == 'W':
        # mid piece belongs to opponent
        return True if (mid_piece == 'b' or mid_piece == 'B') else False
    # white pawn
    elif fr_piece == 'w':
        # mid piece belongs to opponent
        # white pawns only move up
        return True if (mid_piece == 'b' or mid_piece == 'B') and (to_y - fr_y == -2) else False
    # black king
    elif fr_piece == 'B':
        return True if (mid_piece == 'w' or mid_piece == 'W') else False
    # black pawn
    elif fr_piece == 'b':
        # mid piece belongs to opponent
        # black pawns only move down
        return True if (mid_piece == 'w' or mid_piece == 'W') and (to_y - fr_y == 2) else False

def possible_simple_moves(board, fr):
    fr_piece = board[fr]
    fr_y = fr[0]
    fr_x = fr[1]
    targets = []
    # white king
    if fr_piece == 'W':
        # all 4 diagonal directions
        targets = [(fr_y-1, fr_x-1), #up-left
                    (fr_y-1, fr_x+1), #up-right
                    (fr_y+1, fr_x-1), #low-left
                    (fr_y+1, fr_x+1)] #low-right
    # white pawn
    elif fr_piece == 'w':
        # white pawns only move up
        targets = [(fr_y-1, fr_x-1), #up-left
                    (fr_y-1, fr_x+1)] #up-right
    # black king
    elif fr_piece == 'B':
        # all 4 diagonal directions
        targets = [(fr_y-1, fr_x-1), #up-left
                    (fr_y-1, fr_x+1), #up-right
                    (fr_y+1, fr_x-1), #low-left
                    (fr_y+1, fr_x+1)] #low-right
    # black pawn
    elif fr_piece == 'b':
        # black pawns only move down
        targets = [(fr_y+1, fr_x-1), #low-left
                    (fr_y+1, fr_x+1)] #low-right

    result = []
    for t in targets:
        if valid_simple_move(board, fr, t):
            result.append([fr,t])

    return result

def possible_jumps_wrapper(board, fr):
    result = possible_jumps(board, fr)
    return result if len(result[0]) >= 2 else []

# returns a list of lists, each a path till no more jumps possible, includes fr and end
def possible_jumps(board, fr):
    fr_piece = board[fr]
    fr_y = fr[0]
    fr_x = fr[1]
    targets = []
    # white king
    if fr_piece == 'W':
        # all 4 diagonal directions
        targets = [(fr_y-2, fr_x-2), #up-left
                    (fr_y-2, fr_x+2), #up-right
                    (fr_y+2, fr_x-2), #low-left
                    (fr_y+2, fr_x+2)] #low-right
    # white pawn
    elif fr_piece == 'w':
        # white pawns only move up
        targets = [(fr_y-2, fr_x-2), #up-left
                    (fr_y-2, fr_x+2)] #up-right
    # black king
    elif fr_piece == 'B':
        # all 4 diagonal directions
        targets = [(fr_y-2, fr_x-2), #up-left
                    (fr_y-2, fr_x+2), #up-right
                    (fr_y+2, fr_x-2), #low-left
                    (fr_y+2, fr_x+2)] #low-right
    # black pawn
    elif fr_piece == 'b':
        # black pawns only move down
        targets = [(fr_y+2, fr_x-2), #low-left
                    (fr_y+2, fr_x+2)] #low-right

    # contains all valid targets from fr
    branches = []
    for t in targets:
        if valid_jump(board, fr, t):
            branches.append(t)

    if not branches:
        return [[fr]]

    result = []
    for b in branches:
        new_board = copy.deepcopy(board)
        move(new_board, fr, b)
        # returns a list of paths fr b to end inclusive
        b_to_end = possible_jumps(new_board, b)
        for rb in b_to_end:
            # a path from fr to end inclusive
            result.append([fr] + rb)

    return result

# given a list of list of moves, convert to non-repeating one step moves
# returns a list of 4-tuples
def one_step_moves(moves):
    result = []
    for sequence in moves:
        seq_len = len(sequence)
        for i in range(seq_len-1):
            osm = sequence[i] + sequence[i+1]
            if osm not in result:
                result.append(osm)
    return result

class Board:
    def __init__(self, filename='input.txt', califile='calibration.txt'):
        with open(filename, 'r') as f:
            self.mode = f.readline()[:-1]
            self.side = f.readline()[:-1]
            self.time = float(f.readline()[:-1])

            self.board = np.full((8,8), '.')
            self.white_pieces = {'kings':[], 'pawns':[]}
            self.black_pieces = {'kings':[], 'pawns':[]}
            i = 0
            j = 0
            while i <= 7:
                while j <= 7:
                    c = f.read(1)
                    if c == '.':
                        j += 1
                    elif c == 'w':
                        self.board[i,j] = 'w'
                        self.white_pieces['pawns'].append((i,j))
                        j += 1
                    elif c == 'W':
                        self.board[i,j] = 'W'
                        self.white_pieces['kings'].append((i,j))
                        j += 1
                    elif c == 'b':
                        self.board[i,j] = 'b'
                        self.black_pieces['pawns'].append((i,j))
                        j += 1
                    elif c == 'B':
                        self.board[i,j] = 'B'
                        self.black_pieces['kings'].append((i,j))
                        j += 1
                j = 0
                i += 1
        if self.mode == 'GAME':
            self.max_depth = 6
        elif self.mode == 'SINGLE':
            with open('calibration.txt', 'r') as f:
                times_by_depth = list(map(float, f.readline()[:-1].split(',')))
                can_run_depth = 0
                for it, t in enumerate(times_by_depth):
                    if self.time > t:
                        can_run_depth = it+1
                self.max_depth = can_run_depth
        self.white_possible_moves, self.black_possible_moves = self.possible_moves()
        self.game_state = self.game_over()

    # makes a single move and updates pieces
    # in-place
    # assumes valid move
    def move(self, fr, to):
        piece = self.board[fr]
        self.board[fr] = '.'
        mid_y = (to[0]+fr[0])/2.0
        # jump move
        if mid_y.is_integer():
            mid_x = (to[1]+fr[1])/2.0
            mid = (int(mid_y), int(mid_x))
            mid_piece = self.board[mid]
            self.board[mid] = '.'
            # remove captured piece
            if mid_piece == 'w':
                self.white_pieces['pawns'].remove(mid)
            elif mid_piece == 'W':
                self.white_pieces['kings'].remove(mid)
            elif mid_piece == 'b':
                self.black_pieces['pawns'].remove(mid)
            elif mid_piece == 'B':
                self.black_pieces['kings'].remove(mid)
        # simple move
        self.board[to] = piece
        # remove old location fr piece collections and add new location
        # and crown kings
        if piece == 'w':
            self.white_pieces['pawns'].remove(fr)
            # king
            if to[0] == 0:
                self.white_pieces['kings'].append(to)
                self.board[to] = 'W'
            # remain pawn
            else:
                self.white_pieces['pawns'].append(to)
        elif piece == 'W':
            self.white_pieces['kings'].remove(fr)
            self.white_pieces['kings'].append(to)
        elif piece == 'b':
            self.black_pieces['pawns'].remove(fr)
            # king
            if to[0] == 7:
                self.black_pieces['kings'].append(to)
                self.board[to] = 'B'
            # remain pawn
            else:
                self.black_pieces['pawns'].append(to)
        elif piece == 'B':
            self.black_pieces['kings'].remove(fr)
            self.black_pieces['kings'].append(to)

    # takes in a move list
    # in-place
    # assumes valid move
    def seq_move(self, move):
        # one_step_moves takes in a list of lists and returns a list of 4-tuples
        one_steps = one_step_moves([move])
        for p in one_steps:
            self.move((p[0],p[1]), (p[2],p[3]))
        self.side = 'WHITE' if self.side == 'BLACK' else 'BLACK'
        # update possible moves
        self.white_possible_moves, self.black_possible_moves = self.possible_moves()
        self.game_state = self.game_over()

    # returns a new Board object corresponding to post-move state
    def post_move_Board(self, move):
        new_Board = copy.deepcopy(self)
        new_Board.seq_move(move)
        return new_Board

    # returns possible moves on both sides,
    def possible_moves(self):
        white_simple_moves = []
        white_jumps = []
        black_simple_moves = []
        black_jumps = []
        for l in self.white_pieces.values():
            for p in l:
                p_jumpts = possible_jumps_wrapper(self.board, p)
                # no possible jump at p
                if not p_jumpts:
                    # no jump detected so far, search simple moves
                    if not white_jumps:
                        p_simple_moves = possible_simple_moves(self.board, p)
                        white_simple_moves += p_simple_moves
                else:
                    white_jumps += p_jumpts
        for l in self.black_pieces.values():
            for p in l:
                p_jumpts = possible_jumps_wrapper(self.board, p)
                if not p_jumpts:
                    if not black_jumps:
                        p_simple_moves = possible_simple_moves(self.board, p)
                        black_simple_moves += p_simple_moves
                else:
                    black_jumps += p_jumpts

        # if jumps empty return simple moves else jumps
        if not white_jumps:
            if not black_jumps:
                # no jumps for white or black
                return ('E', white_simple_moves), ('E', black_simple_moves)
            else:
                # no jumps for white, but for black
                return ('E', white_simple_moves), ('J', black_jumps)
        else:
            if not black_jumps:
                # no jumps for black, but for white
                return ('J', white_jumps), ('E', black_simple_moves)
            else:
                # jumps for both
                return ('J', white_jumps), ('J', black_jumps)

    def game_over(self):
        white_no_pieces = True if (not self.white_pieces['kings']) and (not self.white_pieces['pawns']) else False
        white_no_moves = True if not self.white_possible_moves[1] else False
        black_won = white_no_pieces or white_no_moves
        black_no_pieces = True if (not self.black_pieces['kings']) and (not self.black_pieces['pawns']) else False
        black_no_moves = True if not self.black_possible_moves[1] else False
        white_won = black_no_pieces or black_no_moves
        # ensure white and black are treated fairly
        if not (white_won and black_won):
            if white_won:
                return 'WHITE'
            elif black_won:
                return 'BLACK'
        return 'CONT'

    def write_best_to(self, filename='output.txt'):
        best_move = self.alpha_beta_search()
        best_move_one_steps = one_step_moves([best_move])
        if self.side == 'WHITE':
            move_type = self.white_possible_moves[0]
        elif self.side == 'BLACK':
            move_type = self.black_possible_moves[0]
        with open(filename, 'w') as f:
            for p in best_move_one_steps:
                line = move_type+' '+output_indices[p[0],p[1]]+' '+output_indices[p[2],p[3]]+'\n'
                f.write(line)

    def alpha_beta_search(self):
        if self.side == 'WHITE':
            if self.max_depth <= 0:
                return random.choice(self.white_possible_moves[1])
            v, iv = self.max_value(0, float('-inf'), float('inf'))
            return self.white_possible_moves[1][iv]
        elif self.side == 'BLACK':
            if self.max_depth <= 0:
                return random.choice(self.black_possible_moves[1])
            v, iv = self.max_value(0, float('-inf'), float('inf'))
            return self.black_possible_moves[1][iv]

    def max_value(self, depth, alpha, beta):
        a = alpha
        if depth >= self.max_depth or self.game_state != 'CONT':
            return self.eval_func(), None
        v = float('-inf')
        iv = None
        if self.side == 'WHITE':
            for im, m in enumerate(self.white_possible_moves[1]):
                new_Board = self.post_move_Board(m)
                eval, _ = new_Board.min_value(depth+1, a, beta)
                if eval > v:
                    v = eval
                    iv = im
                if v >= beta:
                    return v, iv
                a = max(a, v)
        elif self.side == 'BLACK':
            for im, m in enumerate(self.black_possible_moves[1]):
                new_Board = self.post_move_Board(m)
                eval, _ = new_Board.min_value(depth+1, a, beta)
                if eval > v:
                    v = eval
                    iv = im
                if v >= beta:
                    return v, iv
                a = max(a, v)
        return v, iv

    def min_value(self, depth, alpha, beta):
        b = beta
        if depth >= self.max_depth or self.game_state != 'CONT':
            return self.eval_func(), None
        v = float('inf')
        iv = None
        if self.side == 'WHITE':
            for im, m in enumerate(self.white_possible_moves[1]):
                new_Board = self.post_move_Board(m)
                eval, _ = new_Board.max_value(depth+1, alpha, b)
                if eval < v:
                    v = eval
                    iv = im
                if v <= alpha:
                    return v, iv
                b = min(b, v)
        elif self.side == 'BLACK':
            for im, m in enumerate(self.black_possible_moves[1]):
                new_Board = self.post_move_Board(m)
                eval, _ = new_Board.max_value(depth+1, alpha, b)
                if eval < v:
                    v = eval
                    iv = im
                if v <= alpha:
                    return v, iv
                b = min(b, v)
        return v, iv

    # take center while keeping pieces close
    # trade when at advantage
    # leave 2 pieces to defend king row
    def eval_func(self):
        who_won = self.game_state
        if self.side == who_won:
            return 1000
        elif who_won != 'CONT' and self.side != who_won:
            return -1000
        else:
            if self.side == 'WHITE':
                my_piece_count = 2*len(self.white_pieces['kings']) + len(self.white_pieces['pawns'])
                oppo_piece_count = 2*len(self.black_pieces['kings']) + len(self.black_pieces['pawns'])

                my_jump_count = 0
                if self.white_possible_moves[0] == 'J':
                    my_jump_count = len(one_step_moves(self.white_possible_moves[1]))

                oppo_jump_count = 0
                if self.black_possible_moves[0] == 'J':
                    oppo_jump_count = len(one_step_moves(self.black_possible_moves[1]))
            elif self.side == 'BLACK':
                my_piece_count = 2*len(self.black_pieces['kings']) + len(self.black_pieces['pawns'])
                oppo_piece_count = 2*len(self.white_pieces['kings']) + len(self.white_pieces['pawns'])

                my_jump_count = 0
                if self.black_possible_moves[0] == 'J':
                    my_jump_count = len(one_step_moves(self.black_possible_moves[1]))

                oppo_jump_count = 0
                if self.white_possible_moves[0] == 'J':
                    oppo_jump_count = len(one_step_moves(self.white_possible_moves[1]))

            # more jumps is better but we can only take one
            return my_piece_count - oppo_piece_count + 0.5*(my_jump_count - oppo_jump_count)

def write_Board_to(filename='default.txt', Board=None, default=False):
    if default:
        with open(filename, 'w') as f:
            f.write('GAME\n')
            f.write('BLACK\n')
            f.write('1000.0\n')
            for row in basic_board:
                for col in row:
                    f.write(col)
                f.write('\n')
    else:
        with open(filename, 'w') as f:
            f.write(Board.mode+'\n')
            f.write(Board.side+'\n')
            f.write(str(Board.time)+'\n')
            for row in Board.board:
                for col in row:
                    f.write(col)
                f.write('\n')

if __name__ == "__main__":
    runCase = Board()
    runCase.write_best_to()
