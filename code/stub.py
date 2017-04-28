# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        new_action = npr.rand() < 0.1
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


    def _extract_features(self, state):

        score = state['score']
        tree_dist = state['tree']['dist']
        tree_top = state['tree']['top']
        tree_bot = state['tree']['bot']
        monkey_vel = state['monkey']['vel']
        monkey_top = state['monkey']['top']
        monkey_bot = state['monkey']['bot']
        tree_mid = (tree_top - tree_bot)
        monkey_mid = (monkey_top - monkey_bot)
        monkey_below_down = int(tree_mid < monkey_mid and monkey_vel < 0)
        monkey_below_up = int(tree_mid < monkey_mid and monkey_vel > 0)
        monkey_above_down = int(tree_mid > monkey_mid and monkey_vel < 0)
        monkey_above_up = int(tree_mid > monkey_mid and monkey_vel > 0)
        feature_dict = {'score': score,
                        'tree_dist': tree_dist,
                        'tree_top': tree_top,
                        'tree_bot': tree_bot,
                        'monkey_vel': monkey_vel,
                        'monkey_top': monkey_top,
                        'monkey_bot': monkey_bot,
                        'monkey_below_down': monkey_below_down,
                        'monkey_below_up': monkey_below_up,
                        'monkey_above_down': monkey_above_down,
                        'monkey_above_up': monkey_above_up}
        return frozenset(feature_dict.items())


        # gravity = 2 #infer





def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 20, 10)

	# Save history. 
	np.save('hist',np.array(hist))


