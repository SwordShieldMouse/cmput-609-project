from algs import *

env1 = gym.make('CartPole-v0')
env2 = gym.make('MountainCar-v0')
env3 = gym.make("Acrobot-v1")
env4 = gym.make("LunarLander-v2")

# env1 and env4 have episodes that are short length so don't have to artificially terminate them

# consider artificially terminating episodes if they run too long?

envs = [env4, env1]

load_data = False
run_experiment = True
test_run = False
save_data = not(load_data)
episodes = 50
gamma = 0.99
episode_length = None

# learning rates we will try
lrs = [1e-4 * (2 ** i) for i in range(5)]
trials = 50

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
                np.random.seed(609) # the random seed should be reset for every type of experiment since we want fair comparisons between algs
                t_start = time.time()
                env_name = env.unwrapped.spec.id
                print("performing trials for env = {}, lr = {}, use_entropy = {}".format(env_name, lr, use_entropy))
                for trial in range(trials):
                    returns["env"] += [env_name] * episodes
                    returns["lr"] += [lr] * episodes
                    returns["use_entropy"] += [use_entropy] * episodes

                    trial_t_start = time.time()

                    if env == env2 or env == env3:
                        print("will limit episode length to {} time steps".format(episode_length))
                        # if we are doing either of these environments, limit the episode length b/c they take a v long time to run
                        data = train(env = env, lr = lr, gamma = gamma, use_entropy = use_entropy, episodes = episodes, episode_length = episode_length)
                    else:
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
    print("data loaded")
else:
    df = pd.DataFrame(returns)

# save the data
if save_data is True:
    df.to_csv("experiments.csv", index = False)
    print("data saved")

## plot data
# plots needed: Best lr curves for each alg and env, plot of all lr's within algs for each env
# best lr's within alg for each env
for env in envs:
    for use_entropy in (True, False):
        env_name = env.unwrapped.spec.id
        curr_env = df.env == env_name
        curr_entropy = df.use_entropy == use_entropy
        ax = seaborn.lineplot(x = "episode", y = "return", hue = "lr", legend = "full", data = df.loc[curr_env & curr_entropy, :])
        ax.set_title("Learning curves for use_entropy = {} on {}".format(str(use_entropy), env_name))
        plt.savefig("figs\\{}-{}-experiments.png".format(env_name, str(use_entropy)))
        plt.show()


## do statistical tests?
# other ways of evaluating: episodes required to get to a certain reward
