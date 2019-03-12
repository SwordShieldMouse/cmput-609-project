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

def train(env, lr, gamma, use_entropy, episodes = 100, episode_length = None):
    # do REINFORCE because we only have one step-size
    # can do actor-critic later if there is time
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    #print(action_dim, obs_dim)
    policy = Policy(obs_dim, action_dim).to(device)
    sgd = torch.optim.SGD(policy.parameters(), lr = lr)

    returns = []
    for episode in range(episodes):
        obs = env.reset()
        done = False

        # make sure we use the actual rewards to evaluate how we do
        actual_rewards = []

        # stores the rewards to which we might add entropy
        rewards = []

        states = [torch.Tensor(obs).to(device)]
        actions = []
        t = 0 # to index episode time
        while done is not True:
            #env.render()

            log_probs = policy(torch.Tensor(obs).to(device))
            m = Categorical(logits = log_probs)
            #print("time = {}".format(t), log_probs, obs)
            action = m.sample()
            entropy = m.entropy()

            obs, reward, done, info = env.step(action.item())
            actual_rewards.append(reward)
            # also try adding just -\ln p_i for the action that is taken
            if use_entropy == True:
                reward += entropy

            states.append(torch.Tensor(obs).to(device))
            rewards.append(reward)
            actions.append(action.item())

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
        returns.append(sum([(gamma ** i) * actual_rewards[i] for i in range(len(actual_rewards))]))
    return returns
