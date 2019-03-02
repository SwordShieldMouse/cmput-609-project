from includes import *

class Policy(nn.Module):
    # assuming we have a linear policy
    # Q: should the entropy be with respect to the current policy or with respect to a base policy that changes less often?
    def __init__(self, d_state, d_action):
        super(Policy, self).__init__()
        self.layer = nn.Linear(d_state, d_action)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        out = self.softmax(self.layer(x))
        return out

    def get_entropy(self, x):
        # returns the entropy of the policy given the current state
        # don't need grad for this
        with torch.no_grad():
            probs = self.softmax(self.layer(x)).squeeze().tolist()
            return -sum([p * math.log(p) for p in probs])

def train(env, lr, gamma, use_entropy, episodes = 100):
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
        while done is not True:
            env.render()

            probs = policy(torch.Tensor(obs).to(device))
            #entropy = policy.get_entropy(torch.Tensor(obs).to(device))
            m = Categorical(probs)
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
        for i in range(len(actions)):
            if i > len(rewards) - 1:
                G = 0
            else:
                G = sum([(gamma ** j) * rewards[j] for j in range(i, len(rewards))])
            eligibity_vec = torch.log(policy(states[i])[actions[i]]) # evaluate the gradient with the current params
            loss = -G * eligibity_vec
            sgd.zero_grad()
            loss.backward(retain_graph = True)
            sgd.step()
        # calculate total return for this episode
        returns.append(sum([(gamma ** i) * actual_rewards[i] for i in range(len(actual_rewards))]))
    return returns
