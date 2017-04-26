#
#                                   IMPORTANT NOTE:
#  Large parts of this code and structure is based on work Aron and Joe did for their final project in CS 182.
#  Permission for reuse granted by course staff.
#  See https://github.com/josephwandile/flaippy-bird for reference.
#
import numpy as np
from SwingyMonkey import SwingyMonkey
from collections import defaultdict
import os
import pickle
import random

SWING, JUMP = 0, 1


class Learner(object):

    def __init__(self, epsilon=None, import_from=None, export_to=None, exploiting=False, epochs=20):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        self.epsilon = epsilon          # for epsilon-greedy
        self.alpha = 0.8                # learning rate
        self.gamma = 0.7                # discount
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

    @staticmethod
    def _get_bucket(val, size=40):
        return val - (val % size)

    def _extract_features(self, state):

        score = state['score']
        tree_dist = state['tree']['dist']
        tree_top = state['tree']['top']
        tree_bot = state['tree']['bot']
        monkey_vel = state['monkey']['vel']
        monkey_top = state['monkey']['top']
        monkey_bot = state['monkey']['bot']
        tree_mid = tree_bot + (tree_top - tree_bot) / 2
        monkey_mid = monkey_bot + (monkey_top - monkey_bot) / 2
        monkey_to_tree = monkey_mid - tree_mid
        monkey_below_down = int(tree_mid < monkey_mid and monkey_vel < 0)
        monkey_below_up = int(tree_mid < monkey_mid and monkey_vel > 0)
        monkey_above_down = int(tree_mid > monkey_mid and monkey_vel < 0)
        monkey_above_up = int(tree_mid > monkey_mid and monkey_vel > 0)

        vel_indicator = 0
        if monkey_vel <= -5:
            vel_indicator = -1
        elif monkey_vel >= 5:
            vel_indicator = 1

        feature_dict = {  # More granular buckets leads to larger state space, slower convergence.
            'tree_dist': self._get_bucket(tree_dist, size=150),
            'monkey_to_tree': self._get_bucket(monkey_to_tree, size=50),
            # 'monkey_below_down': monkey_below_down,    # Monkey is below the midpoint of the gap and moving downwards
            # 'monkey_above_down': monkey_above_down,    # Monkey is above the midpoint of the gap and moving downwards
            'vel': vel_indicator,
            'close_to_bottom': int(monkey_bot < 100),  # Close to bottom of the screen
            'close_to_top': int(monkey_top > 300),     # Close to top of screen
            'gravity': self.gravity,
        }

        return frozenset(feature_dict.items())

    def _update(self, last_state, last_action, current_state, last_reward):
        q = self._get_q_value(last_state, last_action)
        q_ = (1 - self.alpha) * q + self.alpha * (last_reward + self.gamma * self._get_value(current_state))
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


class ApproximateLearner(Learner):

    def __init__(self, epochs, export_to):
        Learner.__init__(self, epochs=epochs, export_to=export_to)
        self.w = None
        self.alpha = .1
        self.gamma = .7

    def _update(self, last_state, last_action, current_state, last_reward):
        for i, _ in enumerate(self.w):
            self.w[i] = self.w[i] + self.alpha * \
                        (last_reward + self.gamma + self._get_value(current_state) - self._get_q_value(last_state, last_action)) * \
                        current_state[i]

        normalized_weights = self.w / np.max(np.abs(self.w))
        self.w = list(normalized_weights)

    def _get_q_value(self, state, action):
        return np.dot(state, self.w)

    def _extract_features(self, state):

        tree_dist = state['tree']['dist']
        tree_top = state['tree']['top']
        tree_bot = state['tree']['bot']
        monkey_vel = state['monkey']['vel']
        monkey_top = state['monkey']['top']
        monkey_bot = state['monkey']['bot']
        last_action = self.last_action or 0

        features = [
            tree_dist,
            monkey_vel,
            monkey_top,
            monkey_bot,
            self.gravity,
            # 1.0,
            # last_action,
            tree_bot,
            tree_top,
        ]

        if not self.w:
            self.w = list(np.random.uniform(low=-1, size=len(features)))

        return features


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

    epochs = 200

    # Select agent
    # agent = Learner(epochs=epochs, export_to='qs.pkl')
    agent = ApproximateLearner(epochs=epochs, export_to='qs.pkl')

    # Empty list to save history
    hist = []

    # Run games
    run_games(agent, hist, iters=epochs, t_len=0)

    print("High Score: {}".format(np.max(hist)))
    print("Average Score: {}".format(np.mean(hist)))
    print("Average of last {}: {}".format(20, np.mean(hist[-20:])))
    print("Number of States: {}".format(len(agent.q_values)))
