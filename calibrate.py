import cProfile, pstats
from homework import *

def run_with_depth(depth):
    testCase = Board(filename='default.txt')
    testCase.max_depth = depth
    testCase.write_best_to()

with open('calibration.txt', 'w') as f:
    max_depth_tested = 11
    write_Board_to(default=True)
    for depth in range(1,max_depth_tested):
        longest = float('-inf')
        # run more for shallower depths
        for i in range(depth,max_depth_tested+1):
            profiler = cProfile.Profile()
            profiler.enable()
            run_with_depth(depth)
            profiler.disable()
            ps = pstats.Stats(profiler)
            if ps.total_tt > longest:
                longest = ps.total_tt
        f.write(str(longest)+',')
