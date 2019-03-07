from algs import *

env1 = gym.make('CartPole-v0')
env2 = gym.make('MountainCar-v0')
env3 = gym.make("Acrobot-v1")

envs = [env1]

np.random.seed(609)

load_data = False
run_experiment = True
test_run = False
save_data = True
episodes = 100
gamma = 0.99

# learning rates we will try
lrs = [1e-4 * (2 ** i) for i in range(5)]
trials = 10

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
                t_start = time.time()
                env_name = env.unwrapped.spec.id
                print("performing trials for env = {}, lr = {}, use_entropy = {}".format(env_name, lr, use_entropy))
                for trial in range(trials):
                    returns["env"] += [env_name] * episodes
                    returns["lr"] += [lr] * episodes
                    returns["use_entropy"] += [use_entropy] * episodes
                    trial_t_start = time.time()
                    data = train(env = env, lr = lr, gamma = gamma, use_entropy = use_entropy, episodes = episodes)
                    trial_t_end = time.time()
                    returns["episode"] += episode_ixs
                    returns["return"] += data
                    print("trial {} took {}s".format(trial + 1, trial_t_end - trial_t_start))
                    #returns["ts"] = np.average(data, axis = 0)
                    #returns["std"] = np.std(data, axis = 0)
                t_end = time.time()
                print("time to complete: {}s".format(t_end - t_start))

if load_data == True:
    df = pd.read_csv("experiments.csv")
else:
    df = pd.DataFrame(returns)
## plot data
# plots needed: Best lr curves for each alg and env, plot of all lr's within algs for each env
# best lr's within alg for each env
for env in envs:
    for use_entropy in (True, False):
        env_name = env.unwrapped.spec.id
        curr_env = df.env == env_name
        curr_entropy = df.use_entropy == use_entropy
        ax = seaborn.lineplot(x = "episode", y = "return", hue = "lr", legend = "full", data = df.loc[curr_env & curr_entropy, :])
        plt.savefig("figs/{}-{}-experiments.png".format(env, curr_entropy))
        plt.show()

# save the data
if save_data is True:
    df.to_csv("experiments.csv", index = False)

## do statistical tests?
# other ways of evaluating: episodes required to get to a certain reward
