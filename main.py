from algs import *

env1 = gym.make('CartPole-v0')
env2 = gym.make('MountainCar-v0')
env3 = gym.make("Acrobot-v1")

returns = {}
data = train(env = env1, lr = 1e-3, gamma = 0.99, use_entropy = True)
seaborn.lineplot(x = "Episode", y = "Return", data = data)
plt.show()
