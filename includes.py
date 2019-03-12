import gym
import torch
import torch.nn as nn
import numpy as np
import math
from torch.distributions import Categorical
import seaborn
from matplotlib import pyplot as plt
import pandas as pd
import time

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
