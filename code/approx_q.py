from copy import deepcopy
from stub import Learner, JUMP, SWING
import numpy as np


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
            tree_bot,
            tree_top,
        ]

        features = list(features / np.max(np.abs(features)))

        if not self.w[JUMP]:
            self.w[JUMP] = list(np.random.uniform(low=-1, size=len(features)))

        if not self.w[SWING]:
            self.w[SWING] = list(np.random.uniform(low=-1, size=len(features)))

        return features