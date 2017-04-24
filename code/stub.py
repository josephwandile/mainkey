import numpy as np
from SwingyMonkey import SwingyMonkey
from collections import defaultdict
import os
import pickle
import random

FALL, JUMP = 0, 1


class Learner(object):
    """
    This agent jumps randomly.

    TODO Gravity is the value of the second state's velocity entry. Extract into a constant.
    """

    def __init__(self, epsilon=None, import_from=None, export_to=None, exploiting=False):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.epsilon = epsilon          # off-policy rate
        self.alpha = 0.7                # learning rate
        self.gamma = 1.0                # discount
        self.exploiting = exploiting    # set to false is still trying to learn a good policy
        self.gravity = None

        self.import_from = import_from
        self.export_to = export_to
        self.dump_interval = 200
        self.reporting_interval = 5

        self.q_values = None
        self._init_q_values()

        self.actions = [FALL, JUMP]

    def reset(self):
        self.last_state  = None
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
        return self.q_values[str((state, action))]

    def _set_q_value(self, state, action, q_):
        self.q_values[str((state, action))] = q_

    def _get_value(self, state):
        return max([self._get_q_value(state, action) for action in self.actions])

    def _get_greedy_action(self, state):
        return FALL if self._get_q_value(state, FALL) >= self._get_q_value(state, JUMP) else JUMP

    def _get_action(self, state):
        action = random.choice(self.actions) if self._off_policy() else self._get_greedy_action(state)
        return action

    def _extract_state(self, state):
        # TODO
        return state

    def _update(self, last_state, last_action, current_state, last_reward):
        q = self._get_q_value(last_state, last_action)
        q_ = q + self.alpha * (last_reward + self.gamma * self._get_value(current_state) - q)
        self._set_q_value(last_state, last_action, q_)

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        if state['monkey']['vel'] == 0 and not self.gravity:
            self.gravity = state['monkey']['vel']

        state_representation = self._extract_state(state)
        self._update(self.last_state, self.last_action, state, self.last_reward)

        new_action = self._get_action(state_representation)
        new_state = state

        self.last_action = new_action
        self.last_state = new_state

        return self.last_action

    def reward_callback(self, reward):
        """
        This gets called so you can see what reward you get.

        Note to self: action_callback is called after reward_callback
        """
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.

    """
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


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games.
    run_games(agent, hist, 20, 0)

    # Save history.
    np.save('hist', np.array(hist))
