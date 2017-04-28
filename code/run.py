from stub import *
from SwingyMonkey import SwingyMonkey

agent = ExactLearner(epochs=10, import_from='demo.w', exploiting=True)
hist = []
    # Run games
run_games(agent, hist, iters=10, t_len=5)
print "scores", hist