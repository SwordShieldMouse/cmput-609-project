from algs import *
from test_envs import *

env1 = gym.make('CartPole-v0')
env2 = gym.make('MountainCar-v0').env
env3 = gym.make("Acrobot-v1").env
env4 = gym.make("LunarLander-v2")
env5 = SCGrid()

# env1 and env4 have episodes that are short length so don't have to artificially terminate them

# consider artificially terminating episodes if they run too long?

envs = [env1, env5]
pseudorewards = ["none", "entropy", "information_content"]

run_experiment = False
render_env = False
print_return = False
load_data = True
test_run = False
save_data = run_experiment
episodes = 50
gamma = 0.99
episode_length = None

# learning rates we will try
lrs = [1e-4 * (2 ** i) for i in range(4)]
trials = 200

returns = {"env": [], "lr": [], "pseudoreward": [], "episode": [], "return": []}
episode_ixs = [i + 1 for i in range(episodes)]

if test_run == True:
    df = {"episode": [], "return": [], "pseudoreward": [], "lr": []}
    for lr in (2e-4, 4e-4):
        for pseudoreward in pseudorewards:
            print("trials for pseudoreward = {}".format(pseudoreward))
            for trial in range(trials):
                print("Starting trial {}".format(trial + 1))
                data = train(env = env5, lr = lr, gamma = gamma, pseudoreward = pseudoreward, episodes = episodes, render_env = render_env, print_return = print_return, episode_length = episode_length)
                df["pseudoreward"] += [pseudoreward] * episodes
                df["return"] += data
                df["episode"] += episode_ixs
                df["lr"] += [lr] * episodes
    df = pd.DataFrame(df)
    #df.lr = df.lr.astype('category')
    #df.lr.cat.remove_unused_categories() # make sure we don't plot extraneous learning rates
    # seaborn provides bootstrapped confidence intervals
    seaborn.lineplot(x = "episode", y = "return", data = df, hue = "lr", style = 'pseudoreward', legend = 'full')
    plt.show()

### experiment
if run_experiment == True:
    for env in envs:
        for lr in lrs:
            for pseudoreward in pseudorewards:
                np.random.seed(609) # the random seed should be reset for every type of experiment since we want fair comparisons between algs
                t_start = time.time()
                try:
                    env_name = env.unwrapped.spec.id
                except:
                    env_name = env.name
                print("performing trials for env = {}, lr = {}, pseudoreward = {}".format(env_name, lr, pseudoreward))
                for trial in range(trials):
                    returns["env"] += [env_name] * episodes
                    returns["lr"] += [lr] * episodes
                    returns["pseudoreward"] += [pseudoreward] * episodes

                    trial_t_start = time.time()

                    if env == env2 or env == env3:
                        print("will limit episode length to {} time steps".format(episode_length))
                        # if we are doing either of these environments, limit the episode length b/c they take a v long time to run
                        data = train(env = env, lr = lr, gamma = gamma, pseudoreward = pseudoreward, episodes = episodes, episode_length = episode_length, render_env = render_env, print_return = print_return)
                    else:
                        data = train(env = env, lr = lr, gamma = gamma, pseudoreward = pseudoreward, episodes = episodes, render_env = render_env, print_return = print_return)

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
if load_data is True or run_experiment is True:
    for env in envs:
        for pseudoreward in pseudorewards:
            try:
                env_name = env.unwrapped.spec.id
            except:
                env_name = env.name
            curr_env = df.env == env_name
            curr_pseudoreward = df.pseudoreward == pseudoreward
            plt.figure(figsize = (15, 10))
            ax = seaborn.lineplot(x = "episode", y = "return", hue = "lr", legend = "full", data = df.loc[curr_env & curr_pseudoreward, :])
            ax.set_title("Learning curves for pseudoreward = {} on {}".format(pseudoreward, env_name))
            plt.savefig("figs\\{}-{}.png".format(env_name, pseudoreward))
            #plt.show()


## do statistical tests?
# plot best learning curves from each method on one plot
for env in envs:
    try:
        env_name = env.unwrapped.spec.id
    except:
        env_name = env.name
    curr_env = df.env == env_name
    best_none = (df.pseudoreward == "none") & (df.lr == 8e-4)
    if env_name == "ShortCorridorGridworld":
        best_entropy = (df.pseudoreward == "entropy") & (df.lr == 1e-4)
    else:
        best_entropy = (df.pseudoreward == "entropy") & (df.lr == 8e-4)
    best_info = (df.pseudoreward == "information_content") & (df.lr == 8e-4)
    rows = curr_env & (best_none | best_entropy | best_info)
    plt.figure(figsize = (15, 10))
    ax = seaborn.lineplot(x = "episode", y = "return", style = "lr", hue = "pseudoreward", legend = "full", data = df.loc[rows, :])
    ax.set_title("Comparison of algorithms on env = {}".format(env_name))
    plt.savefig("figs\\{}-compare.png".format(env_name))
    #plt.show()
