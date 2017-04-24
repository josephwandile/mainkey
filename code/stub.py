import numpy as np
from SwingyMonkey import SwingyMonkey
from collections import defaultdict
import os
import pickle
import random

SWING, JUMP = 0, 1


class Learner(object):
    """
    This agent jumps randomly.

    TODO Gravity is the value of the second state's velocity entry. Extract into a constant.
    """

    def __init__(self, epsilon=None, import_from=None, export_to=None, exploiting=False, epochs=20):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        self.epsilon = epsilon          # off-policy rate
        self.alpha = 0.7                # learning rate
        self.gamma = 1.0                # discount
        self.exploiting = exploiting    # set to false is still trying to learn a good policy
        self.gravity = None

        self.import_from = import_from
        self.export_to = export_to
        self.epochs = epochs
        self.epoch = 0

        self.q_values = None
        self._init_q_values()

        self.actions = [SWING, JUMP]

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def _init_q_values(self):
        if self.import_from:
            if os.path.isfile(self.import_from):
                with open(self.import_from) as infile:
                    self.q_values = defaultdict(float, pickle.load(infile))
        else:
            self.q_values=defaultdict(float)

    def _dump_q_values(self):
        if not self.export_to:
            return

        with open(self.export_to, 'w') as outfile:
            pickle.dump(self.q_values, outfile)

    def _get_epsilon(self):
        """
        Can use a cooling function here to decrease over time.
        """
        return self.epsilon if not self.epsilon else 0.05

    def _off_policy(self):
        if self.exploiting:
            return False

        return random.random() < self._get_epsilon()

    def _get_q_value(self, state, action):
        return self.q_values[state, action]

    def _set_q_value(self, state, action, q_):
        self.q_values[state, action] = q_

    def _get_value(self, state):
        return max([self._get_q_value(state, action) for action in self.actions])

    def _get_greedy_action(self, state):
        return SWING if self._get_q_value(state, SWING) >= self._get_q_value(state, JUMP) else JUMP

    def _get_action(self, state):
        action = random.choice(self.actions) if self._off_policy() else self._get_greedy_action(state)
        return action

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

        feature_dict = {
            'score': score,
            'tree_dist': tree_dist,
            'tree_top': tree_top,
            'tree_bot': tree_bot,
            'monkey_vel': monkey_vel,
            'monkey_top': monkey_top,
            'monkey_bot': monkey_bot,
            'monkey_below_down': monkey_below_down,
            'monkey_below_up': monkey_below_up,
            'monkey_above_down': monkey_above_down,
            'monkey_above_up': monkey_above_up
        }

        return frozenset(feature_dict.items())

    def _update(self, last_state, last_action, current_state, last_reward):
        q = self._get_q_value(last_state, last_action)
        q_ = q + self.alpha * (last_reward + self.gamma * self._get_value(current_state) - q)
        self._set_q_value(last_state, last_action, q_)

    def action_callback(self, state):

        if state['monkey']['vel'] == 0 and not self.gravity:
            self.gravity = state['monkey']['vel']

        state_representation = self._extract_features(state)

        if self.last_state:
            self._update(self.last_state, self.last_action, state_representation, self.last_reward)

        self.last_action = self._get_action(state_representation)
        self.last_state = state_representation

        self.epoch += 1
        if self.epoch == self.epochs:
            self._dump_q_values()

        return self.last_action

    def reward_callback(self, reward):
        """
        Note to self: action_callback is called after reward_callback

        i.e. state, action, reward, next_state is going to be represented in action_callback
        as last_state, last_action, last_reward, state
        """
        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.

    """
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                      # Don't play sounds.
                             text="Epoch {}".format(ii),       # Display the epoch on screen.
                             tick_length=t_len,                # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()


if __name__ == '__main__':

    epochs = 1

    # Select agent
    agent = Learner(epochs=epochs, export_to='qs.pkl')

    # Empty list to save history
    hist = []

    # Run games
    run_games(agent, hist, iters=epochs, t_len=0)

    # Save history
    np.save('hist', np.array(hist))
