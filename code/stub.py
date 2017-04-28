#
#                                   IMPORTANT NOTE:
#  Large parts of this code and structure is based on work Aron and Joe did for their final project in CS 182.
#  Permission for reuse granted by course staff.
#  See https://github.com/josephwandile/flaippy-bird for reference.
#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from SwingyMonkey import SwingyMonkey
from collections import defaultdict
import os
import pickle
import random
from copy import deepcopy

SWING, JUMP = 0, 1
epoch = 0
epochs_total = 0


class Learner(object):
    def __init__(self, import_from=None, export_to=None, exploiting=False, epochs=20, epsilon=0.02, alpha=0.1,
                 gamma=0.7):

        self.last_state = None
        self.last_action = None
        self.last_reward = None

        self.epsilon = epsilon  # for epsilon-greedy
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount
        self.exploiting = exploiting  # set to false is still trying to learn a good policy

        self.import_from = import_from
        self.export_to = export_to
        self.epochs = epochs

        self.w = None  # Store q values, weights for linear models, etc. Arbitrary storage var.
        self._init_q_values()

        self.actions = [SWING, JUMP]

        self.state_history = []
        self.action_history = []

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def dump_q_values(self):
        if not self.export_to:
            return

        with open(self.export_to, 'w') as outfile:
            pickle.dump(self.w, outfile)

    def _get_epsilon(self):
        """
        Can use a cooling function here to decrease over time.
        """
        return self.epsilon * (1 - float(epoch) / epochs_total) + .001
        # return self.epsilon or 0.02

    def _get_alpha(self):
        """
        Cooling function.
        """
        return self.alpha * (1 - float(epoch) / epochs_total) + .01

    def _off_policy(self):
        if self.exploiting:
            return False

        return random.random() < self._get_epsilon()

    def _get_value(self, state):
        return max([self._get_q_value(state, action) for action in self.actions])

    def _get_greedy_action(self, state):
        return SWING if self._get_q_value(state, SWING) >= self._get_q_value(state, JUMP) else JUMP

    def _get_action(self, state):
        action = not self._get_greedy_action(state) if self._off_policy() else self._get_greedy_action(state)
        return int(action)

    def _get_q_value(self, state, action):
        pass

    def _set_q_value(self, state, action, q_):
        pass

    def _init_q_values(self):
        pass

    def _extract_features(self, state):
        pass

    def _update(self, last_state, last_action, current_state, last_reward):
        pass

    def action_callback(self, state):

        self.state_history.append(state)
        state_representation = self._extract_features(state)

        if self.last_state and not self.exploiting:
            self._update(self.last_state, self.last_action, state_representation, self.last_reward)

        self.last_action = self._get_action(state_representation)
        self.action_history.append(self.last_action)

        self.last_state = state_representation

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
        q_ = (1 - self._get_alpha()) * q + self._get_alpha() * (
        last_reward + self.gamma * self._get_value(current_state))
        self._set_q_value(last_state, last_action, q_)

    def _get_q_value(self, state, action):
        return self.w[state, action]

    def _set_q_value(self, state, action, q_):
        self.w[state, action] = q_

    def _extract_features(self, state):

        gravity = state['gravity']

        tree_dist = state['tree']['dist']

        tree_top = state['tree']['top']
        tree_bot = state['tree']['bot']
        tree_mid = tree_bot + (tree_top - tree_bot) / 2

        monkey_top = state['monkey']['top']
        monkey_bot = state['monkey']['bot']
        monkey_mid = monkey_bot + (monkey_top - monkey_bot) / 2

        relative_y = monkey_mid - tree_mid

        monkey_vel = state['monkey']['vel']

        feature_dict = {
            'gravity': gravity,
            'relative_x': tree_dist // 50,
            'relative_y': relative_y // 20,
            'should_jump': int(monkey_vel < 0 and monkey_mid < tree_mid),
            'should_fall': int(monkey_vel > 0 and monkey_mid > tree_mid),
        }

        return frozenset(feature_dict.items())

    def _init_q_values(self):
        if self.import_from:
            if os.path.isfile(self.import_from):
                with open(self.import_from) as infile:
                    self.w = defaultdict(float, pickle.load(infile))
        else:
            self.w = defaultdict(float)


def run_games(learner, hist, iters=100, t_len=1):
    global epoch
    global epochs_total
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    epochs_total = iters
    for ii in range(iters):
        epoch = ii
        # print ii
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch {}".format(ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

    learner.dump_q_values()


def generate_summary(res, title_details):
    plt.figure()
    moving_average = pd.rolling_mean(res, 10)
    plt.plot(res, label='actual scores')
    plt.plot(moving_average, label='moving average')
    plt.title('Scores with {}'.format(title_details))
    plt.legend(loc='upper left')
    plt.savefig(open('time.png', 'w'))
    plt.figure()
    plt.hist(res.values, bins=6)
    plt.title('Score distribution with {}'.format(title_details))
    plt.savefig(open('dist.png', 'w'))

def experiment(epochs, alpha, gamma, epsilon):
    agent = ExactLearner(epochs=epochs, epsilon=epsilon, alpha=alpha, gamma=gamma, export_to='bestCartesian.w')
    hist = []

    # Run games
    run_games(agent, hist, iters=epochs, t_len=0)
    print("State Space Size: {}".format(len(agent.w)))

    maxtrain = max(hist)
    avgtrain = float(sum(hist))/len(hist)
    np.savetxt('res.csv', hist, delimiter=',')
    generate_summary(pd.Series(hist), 'alpha={}, gamma={}'.format(alpha, gamma))
    pickle.dump(agent.state_history, open('states.pkl', 'wb'))
    hist = []
    agent = ExactLearner(epochs=epochs, epsilon=epsilon, alpha=alpha, gamma=gamma, import_from='bestCartesian.w', exploiting=True)
    run_games(agent, hist, iters=100, t_len=0)
    maxtest =  max(hist)
    avgtest = float(sum(hist)) / len(hist)

    print epochs, alpha, gamma, epsilon, maxtrain, avgtrain, maxtest, avgtest




if __name__ == '__main__':
    """
    To get a feel for how well this works, import one of the existing .pkl files
    and run an agent in exploitation mode. This will follow the maximal policy and
    won't update any of the model's parameters.
    e.g. agent = ExactLearner(epochs=10, import_from='already_trained.pkl', exploiting=True)
    """

    # epochs = [50, 75, 100]
    # alphas= [.01, .05, .1]
    # gammas = [.8,.9]
    # eps = [.001, .01]
    # for ep in epochs:
    #     for a in alphas:
    #         for g in gammas:
    #             for e in eps:
    #                 try:
    #                     experiment(ep, a, g, e)
    #                 except Exception as e:
    #                     try:
    #                         experiment(ep, a, g, e)
    #                     except Exception as e:
    #                         continue

    # experiment(300	,0.05	,0.8	,0.001)
    agent = ExactLearner(epochs=10, import_from='demo.w', exploiting=True)
    hist = []
    # Run games
    run_games(agent, hist, iters=10, t_len=5)



