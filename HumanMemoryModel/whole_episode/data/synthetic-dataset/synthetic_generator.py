import simulator
import numpy as np
import pickle as pkl


np.random.seed(0)

n_episodes = 4000
episode_length = 500
n_items = 50
const_delay = 5
dump_path = './data/efc_%d_%d_%d.pkl' % (n_episodes, episode_length, n_items)
dump_file = open(dump_path, 'wb')

# episode_time_sequence = (np.arange(episode_length) + 1) * const_delay
episode_time_sequence = np.random.randint(1, const_delay*2, episode_length)
cnt = 0
for i in range(episode_time_sequence.shape[0]):
    cnt += 5
    episode_time_sequence[i] += cnt 

item_decay_rates = np.exp(np.random.normal(np.log(0.077), 1, n_items))
env_kwargs = {'n_items': n_items, 'time_sequence': episode_time_sequence, 'const_delay': const_delay}
efc_env = simulator.EFCEnv(item_decay_rates=item_decay_rates, **env_kwargs)

hlr_env = simulator.HLREnv(**env_kwargs)

initial_difficulty_csv = "initial_difficulty.csv"
with open(initial_difficulty_csv, 'r') as f:
    n_0s = [float(x.strip()) for x in f.readline().split(',')]
n_0s = np.asarray(n_0s)
a = 0.049
b = 0.0052
tpprl_env = simulator.TPPRLEnv(n_0s=n_0s, alphas=a, betas=b, **env_kwargs)

env = efc_env
q_total = []
qa_total = []
tlast_total = []
nreps_total = []
for i in range(n_episodes):
    obs = env.reset()
    done = False
    q_episode = []
    qa_episode = []
    tlast_episode = []
    nreps_episode = []
    while not done:
        action = np.random.choice(range(n_items))
        feedback, tlast, nreps, done = env.step(action)
        action += 1
        q_episode.append(action)
        qa_episode.append(action+feedback*n_items)
        tlast_episode.append(tlast)
        nreps_episode.append(nreps)
    q_total.append(q_episode)
    qa_total.append(qa_episode)
    tlast_total.append(tlast_episode)
    nreps_total.append(nreps_episode)

q_total = np.array(q_total).astype(np.int32)
qa_total = np.array(qa_total).astype(np.int32)
tlast_total = np.array(tlast_total).astype(np.float32)
nreps_total = np.array(nreps_total).astype(np.float32)
pkl.dump([q_total, qa_total, tlast_total, nreps_total], dump_file, protocol=4)
