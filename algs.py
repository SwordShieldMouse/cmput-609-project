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
    policy = Policy(obs_dim, action_dim)
    sgd = torch.optim.SGD(policy.parameters(), lr = lr)

    returns = []
    for episode in range(episodes):
        obs = env.reset()
        done = False

        rewards = []
        states = [obs]
        actions = []
        action_log_probs = []
        while done is not True:
            env.render()

            probs = policy(torch.Tensor(obs).to(device))
            #entropy = policy.get_entropy(torch.Tensor(obs).to(device))
            m = Categorical(probs)
            action = m.sample()
            entropy = m.entropy()

            obs, reward, done, info = env.step(action.item())
            if use_entropy == True:
                reward += entropy

            rewards.append(reward)
            actions.append(action)
            action_log_probs.append(m.log_prob(action))
        for i in range(len(action_log_probs)):
            if i > len(rewards) - 1:
                G = 0
            else:
                G = sum([(gamma ** (i - j)) * rewards[j] for j in range(i, len(rewards))])
            discount = gamma ** i
            loss = -discount * G * action_log_probs[i]
            sgd.zero_grad()
            loss.backward(retain_graph = True)
            sgd.step()
        # calculate total return for this episode
        returns.append(sum([(gamma ** i) * rewards[i] for i in range(len(rewards))]))
    return returns
