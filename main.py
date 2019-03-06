from algs import *

env1 = gym.make('CartPole-v0')
env2 = gym.make('MountainCar-v0')
env3 = gym.make("Acrobot-v1")

envs = [env1, env2, env3]

np.random.seed(609)

load_data = False
run_experiment = False
test_run = True
save_data = False
episodes = 50
gamma = 0.99

# learning rates we will try
lrs = [1e-4 * (2 ** i) for i in range(10)]
trials = 2

returns = {"env": [], "lr": [], "use_entropy": [], "episode": [], "return": []}
episode_ixs = [i + 1 for i in range(episodes)]

if test_run == True:
    df = {"episode": [], "return": [], "use_entropy": [], "lr": []}
    for lr in (1e-4, 2e-4, 4e-4):
        for use_entropy in (True, False):
            for trial in range(trials):
                print("Starting trial {}".format(trial + 1))
                data = train(env = env1, lr = lr, gamma = gamma, use_entropy = use_entropy, episodes = episodes)
                df["use_entropy"] += [use_entropy] * episodes
                df["return"] += data
                df["episode"] += episode_ixs
                df["lr"] += [lr] * episodes
    df = pd.DataFrame(df)
    #df.lr = df.lr.astype('category')
    #df.lr.cat.remove_unused_categories() # make sure we don't plot extraneous learning rates
    # seaborn provides bootstrapped confidence intervals
    seaborn.lineplot(x = "episode", y = "return", data = df, hue = "lr", style = 'use_entropy', legend = 'full')
    plt.show()

### experiment
if run_experiment == True:
    for env in envs:
        for lr in lrs:
            for use_entropy in (True, False):
                env_name = env.unwrapped.spec.id
                print("performing trials for env = {}, lr = {}, use_entropy = {}".format(env_name, lr, use_entropy))
                for trial in range(trials):
                    returns["env"].append(env_name)
                    returns["lr"].append(lr)
                    returns["use_entropy"].append(use_entropy)
                    # add index to mark the episodes in the returns dict?
                    data = train(env = env, lr = lr, gamma = gamma, use_entropy = use_entropy, episodes = episodes)
                    returns["episode"] += episode_ixs
                    returns["return"] += data
                    #returns["ts"] = np.average(data, axis = 0)
                    #returns["std"] = np.std(data, axis = 0)

## plot data
# plots needed: Best lr curves for each alg, plot of all lr's within algs
df = pd.DataFrame(returns)
# best lr's within alg
for use_entropy in (True, False):
    ax = seaborn.lineplot(x = "episode", y = "return", hue = "lr", legend = "full", data = df.loc[df.use_entropy == use_entropy, :])
    plt.savefig("{}-experiments.png")
    plt.show()

# save the data
if save_data is True:
    df.to_csv("experiments.csv", index = False)

## do statistical tests?
