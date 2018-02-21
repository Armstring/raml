import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import gym
import random
import numpy as np
torch.manual_seed(2333333)

#####################
GameName = "Copy-v0"
env = gym.make(GameName)
num_char = env.observation_space.n #6

######################
dim_obs = num_char # 0,1,2,3,4,5
dim_output = 4* (num_char-1) # ({0,1}, {0,1}, {0,1,2,3,4})
dim_input = dim_output +  dim_obs 
dim_hidden = 256
tau = 0.1

#################
num_epoch = 4000
num_episode = 40
num_trial = 10
batch_size = num_episode*num_trial
MAX_STEP = 1e5
num_iter = 150
clip_norm = 50.0
adam_reset = True
test_size = 100
lr = 0.01
decay_rate = 0.9999

############################
h0 = torch.zeros(dim_hidden).view(1,-1)
c0 = torch.zeros(dim_hidden).view(1,-1)
p0 = 1.0/ dim_output * torch.ones(dim_output).view(1,-1)

#####################	
def weights_init(layer):
	classname = layer.__class__.__name__
	if classname.find('Conv') != -1:
		layer.weight.data.normal_(0.0, 0.02)
	elif classname.find('Lin') != -1:
		layer.weight.data.normal_(0.0, 0.02)

def weights_xavier_init(layer):
	classname = layer.__class__.__name__
	if classname.find('Conv') != -1:
		layer.weight.data.normal_(0.0, 0.02)
	elif classname.find('Lin') != -1:
		init.xavier_uniform(layer.weight.data)

class netP(nn.Module):
	def __init__(self):
		super(netP, self).__init__()
		self.fc1 = nn.Linear(dim_input, dim_input)
		self.lstm = nn.LSTMCell(dim_input, dim_hidden)
		self.fc2 = nn.Linear(dim_hidden, dim_output)
		self.apply(weights_init)
	def forward(self, input, h, c):
		x = self.fc1(input)
		h, c = self.lstm(x, (h,c))
		output = F.softmax(self.fc2(h))
		return output, h, c

def onehot(ind):
	res = torch.zeros(1,dim_obs)
	res[0][ind] = 1.0
	res = Variable(res, requires_grad = False)
	if use_cuda:
		return res.cuda()
	else:
		return res

def softmax(ll):
	ll = np.asarray(ll)
	ll = np.exp(ll - ll.max())
	return ll/np.sum(ll)

def action_id_to_action(ind):
	move = ind//((num_char-1)*2) 
	remain = ind % ((num_char-1)*2) #mod 10
	write  = remain // (num_char-1)
	content = remain% (num_char-1) #mod 5
	return (move, write, content)

def generate_batch(samplerNet, num_episode, number_trial):
	actions_batch = []
	rewards_batch = []
	obs_batch = []
	for ind in range(num_trial):
		seed_int = random.randrange(10000000)
		for episode in range(num_episode):
			env.seed(seed_int)
			obs = env.reset()
			FLAG = False
			
			h = Variable(h0)
			c = Variable(c0)
			output = Variable(p0)
			net_input = torch.cat(( output, onehot(obs)), dim=1)

			action_list = []
			reward_list = []
			obs_list = []
			while not FLAG:
				obs_list.append(obs)
				output, h,c = samplerNet(net_input, h, c)
				action = int(torch.distributions.Categorical(output).sample().data.cpu())
				action_list.append(action)
				action = action_id_to_action(action)
				obs, reward, FLAG, info = env.step(action)
				reward_list.append(reward)
				net_input = torch.cat((output, onehot(obs)), dim=1)

			actions_batch.append(action_list)
			rewards_batch.append(reward_list)
			obs_batch.append(obs_list)
	return actions_batch, rewards_batch, obs_batch
	
def test_reward(net, size):
	for ind in range(size):
		obs = env.reset()
		h = Variable(h0)
		c = Variable(c0)
		output = Variable(p0)
		reward_list = []
		FLAG = False

		while not FLAG:
			net_input = torch.cat(( output, onehot(obs)), dim=1)
			output, h,c = net(net_input, h, c)
			action = int(torch.distributions.Categorical(output).sample().data.cpu())
			action = action_id_to_action(action)
			obs, reward, FLAG, info = env.step(action)
			reward_list.append(reward)

	return np.mean(reward_list)

def construct_w(rewards_batch, num_episode, num_trial):
	weight_list = np.ndarray([])
	for ind in range(num_trial):
		temp = rewards_batch[ind*num_episode: (ind+1)*num_episode]
		temp_weight = [1.0*np.sum(reward_list)/tau for reward_list in temp ]
		temp_weight = softmax(temp_weight)
		weight_list = np.append(weight_list, temp_weight)
	return weight_list
###############################################################################
###############################################################################
policyNet = netP()
policyNet_sample = netP()
optimizer = optim.Adam(policyNet.parameters(), lr=lr)

use_cuda = torch.cuda.is_available()
if use_cuda:
	policyNet = policyNet.cuda()
	policyNet_sample = policyNet_sample.cuda()
	h0 = h0.cuda()
	c0 = c0.cuda()
	p0 = p0.cuda()

num_step = 0
for epoch in range(num_epoch):
	policyNet_sample.load_state_dict(policyNet.state_dict())
	pre_loss = -100.0

	if adam_reset and epoch % 40==0:
		q = optimizer.param_groups[0]['lr']
		optimizer = optim.Adam(policyNet.parameters(), lr=q)

	for iter in range(3*epoch):
		actions_batch, rewards_batch, obs_batch = generate_batch(policyNet_sample, num_episode, num_trial)
		weight_list = construct_w(rewards_batch, num_episode, num_trial)
		#weight_sum_list = [1.0*np.sum(reward_list)/tau for reward_list in rewards_batch ]
		#weight_list = softmax(weight_sum_list)
		loss_list = []
		
		policyNet.zero_grad()
		for ind in range(batch_size):
			weight = weight_list[ind]
			#seed_int = seeds_batch[ind]
			action_list = actions_batch[ind]
			obs_list = obs_batch[ind]

			h = Variable(h0)
			c = Variable(c0)
			output = Variable(p0)
			for action, obs in zip(action_list, obs_list):
				net_input = torch.cat(( output, onehot(obs)), dim=1)
				output, h, c = policyNet(net_input, h, c)

				loss_list.append(-1.0* weight.item()* torch.log(output[0][action])) #-log(\pi)


		loss = sum(loss_list)/batch_size
		loss.backward()
		torch.nn.utils.clip_grad_norm(policyNet.parameters(), clip_norm)
		optimizer.step()
		optimizer.param_groups[0]['lr'] *= decay_rate
		
		#print((pre_loss, loss.data.cpu()[0]))
		if abs(pre_loss - loss.data.cpu()[0]) < 1e-5:
			break
		else:
			pre_loss = loss.data.cpu()[0]
			
		if num_step % 50==0:
			reward_test = test_reward(policyNet, test_size)
			print("[Reset: %r; Epoch: %d; num_step: %d] Pre loss: %.5f; Test reward: %.3f" %(adam_reset, epoch, num_step, pre_loss, reward_test))
		num_step +=1
	if num_step > MAX_STEP:
		break










