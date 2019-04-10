from includes import *

class Policy(nn.Module):
    # assuming we have a linear policy
    # Q: should the entropy be with respect to the current policy or with respect to a base policy that changes less often?
    def __init__(self, d_state, d_action):
        super(Policy, self).__init__()
        self.layer = nn.Linear(d_state, d_action)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, x):
        out = self.softmax(self.layer(x))
        #assert torch.sum(torch.isnan(out) == 0), "policy output is nan, {}".format(self.layer.weight)
        return out

def train_non_stationary(env, lr, gamma, pseudoreward, episodes = 100, episode_length = None, render_env = False, print_return = False):
    # do REINFORCE because we only have one step-size
    # can do actor-critic later if there is time
    try: # if we are working with gym
        action_dim = env.action_space.n
        obs_dim = env.observation_space.shape[0]
    except: # if we are working with our own environments
        action_dim = env.action_dim
        obs_dim = env.obs_dim
    #print(action_dim, obs_dim)
    policy = Policy(obs_dim, action_dim).to(device)
    sgd = torch.optim.SGD(policy.parameters(), lr = lr)
    #false_rewards = {"episode": [], "false_reward": []}
    #policy_probs = {"timestep": [], "prob": []}
    #plt.figure(figsize = (15, 10))
    returns = []
    for episode in range(episodes):
        obs = env.reset()
        done = False

        # make sure we use the actual rewards to evaluate how we do
        actual_rewards = []

        #false_rewards = []

        # stores the rewards to which we might add entropy
        rewards = []

        states = [torch.Tensor(obs).to(device)]
        actions = []
        t = 0 # to index episode timestep
        while done is not True:
            if render_env is True:
                env.render()

            log_probs = policy(torch.Tensor(obs).to(device))
            m = Categorical(logits = log_probs)
            #print("time = {}".format(t), log_probs, obs)
            action = m.sample()

            # record the action probs
            """policy_probs["timestep"].append(t)
            if action == 0: # records prob of going left
                policy_probs["prob"].append(np.exp(m.log_prob(action).detach().item()))
            else:
                policy_probs["prob"].append(1 - np.exp(m.log_prob(action).detach().item()))"""

            if episode >= episodes // 2:
                # switch the actions at the halfway point
                obs, reward, done, info = env.step(1 - action.item())
            else:
                obs, reward, done, info = env.step(action.item())

            # the real rewards we will use for evaluating the algorithms
            actual_rewards.append(reward)

            states.append(torch.Tensor(obs).to(device))
            actions.append(action.item())

            # adding pseudorewards possibly to help with exploration
            if pseudoreward == "entropy":
                reward += m.entropy().detach() # detach so that we don't take the derivative of this
                #print(m.entropy())
                #false_rewards["episode"].append(episode)
                #false_rewards["false_reward"].append(reward.detach().item())
                #false_rewards.append(reward.detach().item())
            elif pseudoreward == "information_content":
                reward -= m.log_prob(action).detach() # detach so that we don't take the derivative of this
                #false_rewards["episode"].append(episode)
                #false_rewards["false_reward"].append(reward.detach().item())
                #false_rewards.append(reward.detach().item())

            # the pseudoreward where we add entropy or information content
            rewards.append(reward)


            t += 1

            if episode_length != None and t >= episode_length:
                break
        for i in range(len(actions)):
            G = 0
            if i <= len(rewards) - 1:
                G = sum([(gamma ** j) * rewards[j] for j in range(i, len(rewards))])

            # don't need to use log since we already output logits
            elig_vec = policy(states[i])[actions[i]] # evaluate the gradient with the current params

            loss = -G * elig_vec
            #assert np.isnan(G.detach()) == False, "G is nan"
            #assert torch.sum(torch.isnan(elig_vec)) == 0, "elig vec is nan"
            #assert np.isnan(loss.detach()) == False, "loss is nan"
            sgd.zero_grad()
            if i + 1 == len(actions):
                loss.backward()
            else:
                loss.backward(retain_graph = True)
            sgd.step()
        # calculate total return for this episode
        curr_return = sum([(gamma ** i) * actual_rewards[i] for i in range(len(actual_rewards))])
        if print_return is True:
            print("return on episode {} is {}".format(episode + 1, curr_return))
        returns.append(curr_return)
        #seaborn.lineplot(x = "timestep", y = "prob", data = policy_probs, ci = None, legend = "full")


    # print the false rewards
    #false_rewards = pd.DataFrame(false_rewards)
    #seaborn.lineplot(x = "episode", y = "false_reward", data = false_rewards, ci = 95)
    #plt.ylabel("Probability of taking action left")
    #plt.xlabel("Timestep")
    #plt.show()
    return returns


def train(env, lr, gamma, pseudoreward, episodes = 100, episode_length = None, render_env = False, print_return = False):
    # do REINFORCE because we only have one step-size
    # can do actor-critic later if there is time
    try: # if we are working with gym
        action_dim = env.action_space.n
        obs_dim = env.observation_space.shape[0]
    except: # if we are working with our own environments
        action_dim = env.action_dim
        obs_dim = env.obs_dim
    #print(action_dim, obs_dim)
    policy = Policy(obs_dim, action_dim).to(device)
    sgd = torch.optim.SGD(policy.parameters(), lr = lr)
    #false_rewards = {"episode": [], "false_reward": []}
    #policy_probs = {"timestep": [], "prob": []}
    #plt.figure(figsize = (15, 10))
    returns = []
    for episode in range(episodes):
        obs = env.reset()
        done = False

        # make sure we use the actual rewards to evaluate how we do
        actual_rewards = []

        #false_rewards = []

        # stores the rewards to which we might add entropy
        rewards = []

        states = [torch.Tensor(obs).to(device)]
        actions = []
        t = 0 # to index episode timestep
        while done is not True:
            if render_env is True:
                env.render()

            log_probs = policy(torch.Tensor(obs).to(device))
            m = Categorical(logits = log_probs)
            #print("time = {}".format(t), log_probs, obs)
            action = m.sample()

            # record the action probs
            """policy_probs["timestep"].append(t)
            if action == 0: # records prob of going left
                policy_probs["prob"].append(np.exp(m.log_prob(action).detach().item()))
            else:
                policy_probs["prob"].append(1 - np.exp(m.log_prob(action).detach().item()))"""

            obs, reward, done, info = env.step(action.item())

            # the real rewards we will use for evaluating the algorithms
            actual_rewards.append(reward)

            states.append(torch.Tensor(obs).to(device))
            actions.append(action.item())

            # adding pseudorewards possibly to help with exploration
            if pseudoreward == "entropy":
                reward += m.entropy().detach() # detach so that we don't take the derivative of this
                #print(m.entropy())
                #false_rewards["episode"].append(episode)
                #false_rewards["false_reward"].append(reward.detach().item())
                #false_rewards.append(reward.detach().item())
            elif pseudoreward == "information_content":
                reward -= m.log_prob(action).detach() # detach so that we don't take the derivative of this
                #false_rewards["episode"].append(episode)
                #false_rewards["false_reward"].append(reward.detach().item())
                #false_rewards.append(reward.detach().item())

            # the pseudoreward where we add entropy or information content
            rewards.append(reward)


            t += 1

            if episode_length != None and t >= episode_length:
                break
        for i in range(len(actions)):
            G = 0
            if i <= len(rewards) - 1:
                G = sum([(gamma ** j) * rewards[j] for j in range(i, len(rewards))])

            # don't need to use log since we already output logits
            elig_vec = policy(states[i])[actions[i]] # evaluate the gradient with the current params

            loss = -G * elig_vec
            #assert np.isnan(G.detach()) == False, "G is nan"
            #assert torch.sum(torch.isnan(elig_vec)) == 0, "elig vec is nan"
            #assert np.isnan(loss.detach()) == False, "loss is nan"
            sgd.zero_grad()
            if i + 1 == len(actions):
                loss.backward()
            else:
                loss.backward(retain_graph = True)
            sgd.step()
        # calculate total return for this episode
        curr_return = sum([(gamma ** i) * actual_rewards[i] for i in range(len(actual_rewards))])
        if print_return is True:
            print("return on episode {} is {}".format(episode + 1, curr_return))
        returns.append(curr_return)
        #seaborn.lineplot(x = "timestep", y = "prob", data = policy_probs, ci = None, legend = "full")


    # print the false rewards
    #false_rewards = pd.DataFrame(false_rewards)
    #seaborn.lineplot(x = "episode", y = "false_reward", data = false_rewards, ci = 95)
    #plt.ylabel("Probability of taking action left")
    #plt.xlabel("Timestep")
    #plt.show()
    return returns
