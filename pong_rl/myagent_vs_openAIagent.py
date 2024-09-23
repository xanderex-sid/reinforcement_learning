import numpy as np
import pickle
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
H = 200  # number of hidden layer neurons
gamma = 0.99  # discount factor for reward

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(80 * 80 * 1, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        # Ensure the input is in float format
        x = x.float()
        # mlp layers
        x = F.relu(self.fc1(x))  # Hidden layer (200)
        x = torch.sigmoid(self.fc2(x))  # Single output with sigmoid
        return x

# Load pre-trained model
model_file = 'saveMLP_v1.p'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Helper functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid function

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[0]
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(float).ravel()

def prepro_mlp(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2 --- (80, 80)
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1 -- still of shape (80, 80)
  return torch.tensor(I.astype(float).ravel()).unsqueeze(0)

def policy_forward(x):
    """Forward pass to get action probability from model."""
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p  # Probability of moving up

# Setup the environment
env = gym.make('ALE/Pong-v5', render_mode='human')
observation = env.reset()[0]
prev_x = None  # Used to calculate the difference between frames

while True:
    env.render()
    # Preprocess the observation
    cur_x = prepro_mlp(observation)
    x = cur_x - prev_x if prev_x is not None else torch.zeros_like(cur_x) #np.zeros(80 * 80)
    prev_x = cur_x

    # Use the pre-trained model to decide an action
    aprob = model(x) #policy_forward(x)
    action = 2 if np.random.uniform() < aprob.item() else 3  # Move up or down (action 2 = UP, action 3 = DOWN)
    # print(env.step(action))
    # Take the action and observe the result
    observation, reward, done, _, info = env.step(action)
    if done:
        observation = env.reset()[0]
        prev_x = None  # Reset frame difference
