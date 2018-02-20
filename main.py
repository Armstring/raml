import numpy as np
import math

import matplotlib
# Force matplotlib to not use and Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import torch

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def policy_net(obs_dim, output_dim, hidden_dim):
