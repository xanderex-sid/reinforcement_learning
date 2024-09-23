import pickle

import numpy as np
import torch
import gym

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.95 # discount factor for reward
resume = False # resume from previous checkpoint?
render = False

device = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(80 * 80 * 1, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        # Ensure the input is in float format
        x = x.float()
        x = F.relu(self.fc1(x))  # Hidden layer (200)
        x = torch.sigmoid(self.fc2(x))  # Single output with sigmoid
        return x

def prepro_mlp(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2 --- (80, 80)
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1 -- still of shape (80, 80)
  return torch.tensor(I.astype(float).ravel()).unsqueeze(0)

def preprocess_CNN(I): # if you want to create a CNN for policy network use this preprocessing
  """ prepro 210x160x3 uint8 frame into torch (80x80) vector """
  I = I[35:195] # crop
  I = I[:,:,0] # Only take first channel
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  I = torch.tensor(I.astype(float)).unsqueeze(0).unsqueeze(0) # (1, 1, 160, 160) shape
  return I

def discount_rewards_v1(r):
    """ take torch tensor of shape (batch, reward) and compute discounted reward """
    discounted_r = torch.zeros_like(r).to(device)
    running_add = 0.0
    for t in reversed(range(0, r.size(0))):
        if r[t] != 0.0: running_add = 0.0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# starting environment
env = gym.make("ALE/Pong-v5")
observation = env.reset()
prev_x = None # used in computing the difference frame
running_reward = None
reward_sum = 0
episode_number = 0

# defining model
if resume: model = pickle.load(open('./saveMLP_v1.p', 'rb'))
else: model = MLP()
model.to(device)
bce_loss = nn.BCELoss(reduction='none') # reduction='none' will calculate BCE loss element-wise
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}") # calculate total parameters

cnt_matches = 0
while True:
    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro_mlp(observation)
    x = cur_x - prev_x if prev_x is not None else torch.zeros_like(cur_x) # (1, 1, 160, 160)
    if prev_x is None: xs = x
    else: xs = torch.cat((xs, x), dim=0) # creating batch of input of a each match # if our AI looses all matches then 1 episode = 21 match. why ??

    # forward the model to get the probability of y = 1
    x = x.to(device)
    aprob = model(x)
    action = 2 if np.random.uniform() < aprob.item() else 3 # roll the dice!
    # store model output
    if prev_x is None: batch_probs = aprob
    else: batch_probs = torch.cat((batch_probs, aprob), dim=0).to(device) # (batch, probs)

    # y is "fake label" -- considering this as a label for this observation --- to move "UP" y = 1
    y = 1 if action == 2 else 0
    y = torch.tensor([y], dtype=torch.float).unsqueeze(0).to(device) # (batch, target)
    if prev_x is None: ys = y
    else: ys = torch.cat((ys, y), dim=0).to(device) # creating batch of target of a each match -- (batch, target)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action) # done=True when episode will end, an episode equals to a full game played until one wins 21 matches.
    reward_sum += reward

    # storing reward of each match
    if reward != 0: cnt_matches += 1
    reward = torch.tensor([reward], dtype=torch.float).unsqueeze(0).to(device) # (batch, reward)
    if prev_x is None: rs = reward
    else: rs = torch.cat((rs, reward), dim=0).to(device) # creating batch of rewards of a each match -- (batch, target)

    prev_x = cur_x

    if done: # one episode i.e. one game is finished
        episode_number += 1

        discounted_rs = discount_rewards_v1(rs) # rs -- torch.size((batch, rewards))
        discounted_rs -= torch.mean(discounted_rs)
        discounted_rs /= torch.std(discounted_rs) # discounted_rs like weights for loss of each batch

        # Compute the loss for each element
        loss_per_element = bce_loss(batch_probs, ys) # e.g. batch_probs - shape: (30, 1), ys - shape: (30, 1), loss_per_element - (30, 1)
        weighted_loss = loss_per_element * discounted_rs.to(device) # e.g. discounted_rs - shape: (30, 1), weighted_loss - (30, 1)
        loss = weighted_loss.mean()

        # Backward pass
        loss.backward() # store the calculated gradient of each parameter

        if episode_number % batch_size == 0: # update parameters of model when episode is multiple of batch_size
            # update weights of the model
            optimizer.step()  # Update weights
            optimizer.zero_grad()  # Reset gradients, as for batch_size (e.g. 10), gradient of each episode was accumulating.
            print("loss :", loss.item())

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        print(f'total matches played in this episode: {cnt_matches}, Note: To change episode, one player should win 21 matches.')


        if episode_number % 5 == 0: pickle.dump(model, open('./saveMLP_v2.p', 'wb'))

        if episode_number  == 200:
            break

        reward_sum = 0
        cnt_matches = 0
        observation = env.reset() # reset env
        prev_x = None
        xs = None
        rs = None
        ys = None
        batch_probs = None


    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        print('ep %d: (match %d) game finished, reward: %f' % (episode_number, cnt_matches, reward) + ('' if reward == -1 else ' !!!!!!!!'))
