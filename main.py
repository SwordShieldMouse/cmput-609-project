from algs import *

env1 = gym.make('CartPole-v0')
env2 = gym.make('MountainCar-v0')
env3 = gym.make("Acrobot-v1")

envs = [env1, env2, env3]

np.random.seed(609)

# learning rates we will try
lrs = [1e-4 * (2 ** i) for i in range(10)]
trials = 200

returns = {}

data = []
for trial in range(trials):
    print("Starting trial {}".format(trial + 1))
    data.append(train(env = env1, lr = 1e-3, gamma = 0.99, use_entropy = True, episodes = 1000))
data = np.average(data, axis = 0)
std = np.std(data, axis = 0)
seaborn.lineplot(data, range(len(data)))
plt.show()
