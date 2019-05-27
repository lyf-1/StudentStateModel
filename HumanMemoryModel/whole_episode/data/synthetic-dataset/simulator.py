import numpy as np
import copy


class StudentEnv:
    def __init__(self, n_items=30, time_sequence=None, const_delay=5):
        self.n_items = n_items
        self.time_sequence = np.concatenate((np.array([0]), time_sequence))
        self.episode_length = time_sequence.shape[0]
        self.const_delay = const_delay

        self.curr_step = 0
        self.prev_memory = 0
        self.count_array = np.zeros(self.n_items)
        self.success_count_array = np.zeros(self.n_items)
        self.fail_count_array = np.zeros(self.n_items)
        self.last_recall_time_array = np.zeros(self.n_items)

        self.action_dim = n_items
        self.observation_dim = n_items * 4

        self.now = None

    def get_dims(self):
        return [self.observation_dim, self.action_dim]

    def _recall_probabilities(self):
        raise NotImplementedError

    def rew(self):
        # now_memory = np.where(self._recall_probabilities() > 0.5)[0].shape[0] / self.n_items
        now_memory = self._recall_probabilities().mean()
        diff_reward = now_memory - self.prev_memory
        self.prev_memory = now_memory
        return diff_reward

    def _update_model(self, item, timestamp, outcome):
        raise NotImplementedError

    def step(self, action):
        if action < 0 or action >= self.n_items:
            raise ValueError

        self.now = self.time_sequence[self.curr_step]
        curr_outcome = 1 if 0.5 <= self._recall_probabilities()[action] else 0
        # curr_outcome = 1 if np.random.random() < self._recall_probabilities()[action] else 0
        tlast = self.last_recall_time_array[action]
        self._update_model(action, self.now, curr_outcome)

        self.curr_step += 1

        rwd = self.rew()
        done = self.curr_step == self.time_sequence.shape[0]
        obs = self.encode_obs(action, curr_outcome)
        info = [self.prev_memory, curr_outcome]
        # return obs, rwd, done, info
        return curr_outcome, float(tlast)/self.const_delay, self.count_array[action], done

    def encode_obs(self, item, outcome):
        self.count_array[item] += 1
        if outcome:
            self.success_count_array[item] += 1
        else:
            self.fail_count_array[item] += 1
        self.last_recall_time_array[item] = self.now
        return np.concatenate(
            (self.success_count_array, self.fail_count_array, self.count_array,
             (self.now - self.last_recall_time_array + self.const_delay)/self.const_delay))

    def init_param(self):
        self.curr_step = 0
        self.prev_memory = 0
        self.count_array = np.zeros(self.n_items)
        self.success_count_array = np.zeros(self.n_items)
        self.fail_count_array = np.zeros(self.n_items)
        self.last_recall_time_array = np.zeros(self.n_items)

    def reset(self):
        self.init_param()
        feedback, tlast, nreps, done = self.step(0)
        # return obs


class EFCEnv(StudentEnv):
    def __init__(self, item_decay_rates=None, **kwargs):
        super(EFCEnv, self).__init__(**kwargs)

        if item_decay_rates is None:
            self.item_decay_rates = np.exp(np.random.normal(np.log(0.077), 1, self.n_items))
        else:
            self.item_decay_rates = item_decay_rates

        self.tlasts = None
        self.strengths = None
        self.init_tlasts = -np.exp(np.random.normal(4.5, 0.5, self.n_items))
        self._init_params()

    def _init_params(self):
        self.tlasts = copy.deepcopy(self.init_tlasts)
        self.strengths = np.ones(self.n_items)

    def _recall_probabilities(self):
        return np.exp(-self.item_decay_rates * (self.now - self.tlasts) / self.strengths)

    def _update_model(self, item, timestamp, outcome):
        self.strengths[item] += 1
        self.tlasts[item] = timestamp

    def reset(self):
        self._init_params()
        return super(EFCEnv, self).reset()


class HLREnv(StudentEnv):
    '''exponential forgetting curve with log-linear memory strength'''

    def __init__(self, loglinear_coeffs=None, **kwargs):
        super(HLREnv, self).__init__(**kwargs)
        self.pseudo_difficulties = None
        if loglinear_coeffs is None:
            coeffs = np.array([1, 1, 0])
            self.pseudo_difficulties = np.random.normal(0, 1, self.n_items)
            self.loglinear_coeffs = np.concatenate((coeffs, self.pseudo_difficulties))
        else:
            self.loglinear_coeffs = loglinear_coeffs
        assert self.loglinear_coeffs.size == 3 + self.n_items
        # print(self.pseudo_difficulties)
        self.tlasts = None
        self.loglinear_feats = None
        # self.init_tlasts = np.exp(np.random.normal(0, 1, self.n_items))
        self.init_tlasts = -np.exp(np.random.normal(4.5, 0.5, self.n_items))
        self._init_params()

    def _init_params(self):
        # self.tlasts = np.ones(self.n_items) * -sys.maxsize
        self.tlasts = copy.deepcopy(self.init_tlasts)
        self.loglinear_feats = np.zeros((self.n_items, 3))  # n_attempts, n_correct, n_incorrect
        self.loglinear_feats = np.concatenate((self.loglinear_feats, np.eye(self.n_items)), axis=1)

    def _strengths(self):
        return np.exp(np.einsum('j,ij->i', self.loglinear_coeffs, self.loglinear_feats))

    def _recall_probabilities(self):
        return np.exp(-(self.now - self.tlasts) / self._strengths())

    def _update_model(self, item, timestamp, outcome):
        self.loglinear_feats[item, 0] += 1
        self.loglinear_feats[item, 1 if outcome == 1 else 2] += 1
        self.tlasts[item] = timestamp

    def reset(self):
        self._init_params()
        return super(HLREnv, self).reset()

        
class TPPRLEnv(StudentEnv):
    def __init__(self, n_0s=None, alphas=0.01, betas=0.01, **kwargs):
        super(TPPRLEnv, self).__init__(**kwargs)
        self.n_0s = n_0s
        self.alphas = alphas
        self.betas = betas
        self.ns = None
        self.tlasts = None
        self.init_tlasts = -np.exp(np.random.normal(4.5, 0.5, self.n_items))
        self._init_params()

    def _init_params(self):
        if self.n_0s is None:
            self.ns = np.exp(np.random.normal(np.log(0.037), 1, self.n_items))
        else:
            self.ns = copy.deepcopy(self.n_0s)
        self.tlasts = copy.deepcopy(self.init_tlasts)

    def _recall_probabilities(self):
        return np.exp(-self.ns * (self.now - self.tlasts))

    def _update_model(self, item, timestamp, outcome):
        if outcome:
            self.ns[item] = self.ns[item] * (1 - self.alphas)
        else:
            self.ns[item] = self.ns[item] * (1 + self.betas)
        self.tlasts[item] = timestamp

    def reset(self):
        self._init_params()
        return super(TPPRLEnv, self).reset()
