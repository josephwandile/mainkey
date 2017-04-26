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
from copy import deepcopy

SWING, JUMP = 0, 1


class Learner(object):

    def __init__(self, epsilon=None, import_from=None, export_to=None, exploiting=False, epochs=20, alpha=0.8, gamma=0.7):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        self.epsilon = epsilon          # for epsilon-greedy
        self.alpha = alpha              # learning rate
        self.gamma = gamma              # discount
        self.exploiting = exploiting    # set to false is still trying to learn a good policy
        self.gravity = None

        self.import_from = import_from
        self.export_to = export_to
        self.epochs = epochs
        self.epoch = 0

        self.w = None  # Store q values, weights for linear models, etc. Arbitrary storage var.
        self._init_q_values()

        self.actions = [SWING, JUMP]

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def _get_epsilon(self):
        """
        Can use a cooling function here to decrease over time.
        """
        return self.epsilon or 0.05

    def _off_policy(self):
        if self.exploiting:
            return False

        return random.random() < self._get_epsilon()

    def _get_q_value(self, state, action):
        pass

    def _set_q_value(self, state, action, q_):
        pass

    def _init_q_values(self):
        pass

    def _dump_q_values(self):
        pass

    def _get_value(self, state):
        return max([self._get_q_value(state, action) for action in self.actions])

    def _get_greedy_action(self, state):
        return SWING if self._get_q_value(state, SWING) >= self._get_q_value(state, JUMP) else JUMP

    def _get_action(self, state):
        action = random.choice(self.actions) if self._off_policy() else self._get_greedy_action(state)
        return action

    def _extract_features(self, state):
        pass

    def _update(self, last_state, last_action, current_state, last_reward):
        pass

    def action_callback(self, state):

        if state['monkey']['vel'] == 0 and not self.gravity:
            self.gravity = state['monkey']['vel']

        state_representation = self._extract_features(state)

        if self.last_state and not self.exploiting:
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


class ExactLearner(Learner):

    def _update(self, last_state, last_action, current_state, last_reward):
        q = self._get_q_value(last_state, last_action)
        q_ = (1 - self.alpha) * q + self.alpha * (last_reward + self.gamma * self._get_value(current_state))
        self._set_q_value(last_state, last_action, q_)

    def _get_q_value(self, state, action):
        return self.w[state, action]

    def _set_q_value(self, state, action, q_):
        self.w[state, action] = q_

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
        if 3 <= monkey_vel <= 6:
            vel_indicator = 1
        elif -6 <= monkey_vel <= -3:
            vel_indicator = -1
        elif monkey_vel <= -7:
            vel_indicator = -2
        elif monkey_vel >= 7:
            vel_indicator = 2

        feature_dict = {  # More granular buckets leads to larger state space, slower convergence.
            'tree_dist': self._get_bucket(tree_dist, size=100),
            'monkey_to_tree': self._get_bucket(monkey_to_tree, size=25),
            # 'monkey_below_down': monkey_below_down,    # Monkey is below the midpoint of the gap and moving downwards
            # 'monkey_above_down': monkey_above_down,    # Monkey is above the midpoint of the gap and moving downwards
            'close_to_bottom': int(monkey_bot < 100),  # Close to bottom of the screen
            'close_to_top': int(monkey_top > 300),     # Close to top of screen
            'gravity': self.gravity,
            'vel': vel_indicator,
        }

        return frozenset(feature_dict.items())

    def _init_q_values(self):
        if self.import_from:
            if os.path.isfile(self.import_from):
                with open(self.import_from) as infile:
                    self.w = defaultdict(float, pickle.load(infile))
        else:
            self.w = defaultdict(float)

    def _dump_q_values(self):
        if not self.export_to:
            return

        with open(self.export_to, 'w') as outfile:
            pickle.dump(self.w, outfile)


class ApproximateLearner(Learner):

    def _update(self, last_state, last_action, current_state, last_reward):

        new_weights = deepcopy(self.w[last_action])

        for i in range(len(new_weights)):
            new_weights[i] = \
                new_weights[i] + self.alpha * current_state[i] * \
                (last_reward + self.gamma + self._get_value(current_state) - self._get_q_value(last_state, last_action))

        self.w[last_action] = list(new_weights / np.max(np.abs(new_weights)))

    def _get_q_value(self, state, action):
        return np.dot(state, self.w[action])

    def _init_q_values(self):
        self.w = [None, None]

    def _extract_features(self, state):

        tree_dist = state['tree']['dist']
        tree_top = state['tree']['top']
        tree_bot = state['tree']['bot']
        monkey_vel = state['monkey']['vel']
        monkey_top = state['monkey']['top']
        monkey_bot = state['monkey']['bot']

        features = [
            tree_dist,
            monkey_vel,
            monkey_top,
            monkey_bot,
            self.gravity,
            tree_bot,
            tree_top,
        ]

        if not self.w[JUMP]:
            self.w[JUMP] = list(np.random.uniform(low=-1, size=len(features)))

        if not self.w[SWING]:
            self.w[SWING] = list(np.random.uniform(low=-1, size=len(features)))

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

    """
    To get a feel for how well this works, import one of the existing .pkl files
    and run an agent in exploitation mode. This will follow the maximal policy and
    won't update any of the model's parameters.

    e.g. agent = ExactLearner(epochs=10, import_from='already_trained.pkl', exploiting=True)
    """

    # Select agent
    # epochs = 100
    # agent = ExactLearner(epochs=epochs, epsilon=0.02, alpha=0.7, gamma=0.7)

    epochs = 30
    agent = ApproximateLearner(epochs=epochs)

    # Empty list to save history
    hist = []

    # Run games
    run_games(agent, hist, iters=epochs, t_len=0)

    print("High Score: {}".format(np.max(hist)))
    print("Average Score: {}".format(np.mean(hist)))
    print("Average of last {}: {}".format(20, np.mean(hist[-20:])))
    print("Number of States / Weights: {}".format(len(agent.w)))
