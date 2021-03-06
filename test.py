import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import gym
import random
import numpy as np
import math
torch.manual_seed(2333333)

#####################
GameName = "Reverse-v0"
env = gym.make(GameName)
num_char = env.observation_space.n #6

######################
dim_obs = num_char # 0,1,2,3,4,5
dim_output = 4* (num_char-1) # ({0,1}, {0,1}, {0,1,2,3,4})
dim_input = dim_output +  dim_obs 
dim_hidden = 64
tau = 0.1
online = False

#################
num_epoch = 2500
num_episode = 10
num_trial = 40
batch_size = num_episode*num_trial
MAX_STEP = 1e5
clip_norm = 50.0
adam_reset = True
test_size = 100
lr = 0.01
decay_rate = 0.9999

############################
h0 = torch.zeros(dim_hidden).view(1,-1)
c0 = torch.zeros(dim_hidden).view(1,-1)

#####################	
def weights_init(layer):
	classname = layer.__class__.__name__
	if 'LSTM' in classname:
		for parameter in layer.parameters():
			parameter.data.normal_(0.0, 0.2)
			parameter.data.clamp_(-2.0,2.0)
	elif 'Lin' in classname:
		layer.weight.data.normal_(0.0, 0.2)
		layer.weight.data.clamp_(-2.0,2.0)

def weights_xavier_init(layer):
	classname = layer.__class__.__name__
	if 'LSTM' in classname:
		for parameter in layer.parameters():
			init.xavier_uniform(parameter.data)
			parameter.data.clamp_(-2.0,2.0)
	elif 'Lin' in classname:
		init.xavier_uniform(layer.weight.data)
		layer.weight.data.clamp_(-2.0,2.0)

class netP(nn.Module):
	def __init__(self):
		super(netP, self).__init__()
		self.fc1 = nn.Linear(dim_input, dim_hidden)
		self.lstm = nn.LSTMCell(dim_hidden, dim_hidden)
		self.fc2 = nn.Linear(dim_hidden, dim_output)
		self.apply(weights_init)
	def forward(self, input, h, c):
		x = self.fc1(input)
		h, c = self.lstm(x, (h,c))
		output = F.softmax(self.fc2(h))
		return output, h, c

def onehot_obs(ind):
	res = torch.zeros(1,dim_obs)
	res[0][ind] = 1.0
	if use_cuda:
		res =  res.cuda()
	res = Variable(res, requires_grad = False)
	return res

def onehot_action(ind):
	res = torch.zeros(1,dim_output)
	if ind>=0:
		res[0][ind] = 1.0
	if use_cuda:
		res =  res.cuda()
	res = Variable(res, requires_grad = False)
	return res

def save_log(var):
	return torch.log(torch.clamp(var,min=1e-10))

def softmax(ll):
	ll = np.asarray(ll)
	ll = np.exp(ll - ll.max())
	ll = 1.0*ll/np.sum(ll)
	return ll

def action_id_to_action(ind):
	move = ind//((num_char-1)*2) 
	remain = ind % ((num_char-1)*2) #mod 10
	write  = remain // (num_char-1)
	content = remain% (num_char-1) #mod 5
	return (move, write, content)

def generate_batch(samplerNet, num_episode, num_trial):
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
			action=-1

			action_list = []
			reward_list = []
			obs_list = []
			while not FLAG:
				net_input = torch.cat(( onehot_action(action), onehot_obs(obs)), dim=1)
				obs_list.append(obs)
				output, h,c = samplerNet(net_input, h, c)
				action = int(torch.distributions.Categorical(output).sample().data.cpu())
				#action = int(torch.max(output, 1)[1].data.cpu()) #argmax
				action_list.append(action)
				action_tuple = action_id_to_action(action)
				obs, reward, FLAG, info = env.step(action_tuple)
				reward_list.append(reward)

			actions_batch.append(action_list)
			rewards_batch.append(reward_list)
			obs_batch.append(obs_list)
	return actions_batch, rewards_batch, obs_batch
	
def test_reward(net, size):
	reward_list = []
	for ind in range(size):
		obs = env.reset()
		h = Variable(h0)
		c = Variable(c0)
		reward_accu = 0.0
		loss_list = []
		FLAG = False
		action = -1
		while not FLAG:
			net_input = torch.cat(( onehot_action(action), onehot_obs(obs)), dim=1)
			output, h,c = net(net_input, h, c)
			action = int(torch.distributions.Categorical(output).sample().data.cpu())
			#action = int(torch.max(output, 1)[1].data.cpu()) #argmax
			action_tuple = action_id_to_action(action)
			obs, reward, FLAG, info = env.step(action_tuple)
			reward_accu += reward
			loss = -1.0* weight.item()* save_log(output[0][y])#-log(\pi)
			loss_list.append(loss) 

		reward_list.append(reward_accu)
	return np.mean(reward_list), -1.0*np.mean(loss_list)

def construct_w(weight_list, num_episode, num_trial):
	res = np.ndarray(0)
	for ind in range(num_trial):
		temp = weight_list[(ind*num_episode):((ind+1)*num_episode)]
		temp_weight = softmax(temp)
		res = np.append(res, temp_weight)
	return res
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

num_step = 0
for epoch in range(1, num_epoch+1):
	if not online:
		policyNet_sample.load_state_dict(policyNet.state_dict())
	pre_loss = -100.0

	if adam_reset and epoch % 50==0:
		q = optimizer.param_groups[0]['lr']
		optimizer = optim.Adam(policyNet.parameters(), lr=q)
	
	for iter in range(int(math.floor(math.sqrt(epoch)))): #while True:	#
		if online:
			actions_batch, rewards_batch, obs_batch = generate_batch(policyNet, num_episode, num_trial)
		else:
			actions_batch, rewards_batch, obs_batch = generate_batch(policyNet_sample, num_episode, num_trial)
		weight_sum_list = [1.0*np.sum(reward_list)/tau for reward_list in rewards_batch ]
		weight_list = construct_w(weight_sum_list, num_episode, num_trial)
		loss_list = []
		
		policyNet.zero_grad()
		for ind in range(batch_size):
			weight = weight_list[ind]
			action_list = actions_batch[ind]
			obs_list = obs_batch[ind]

			h = Variable(h0)
			c = Variable(c0)
			action = -1

			for y, obs in zip(action_list, obs_list):
				net_input = torch.cat(( onehot_action(action), onehot_obs(obs)), dim=1)
				output, h, c = policyNet(net_input, h, c)
				loss = -1.0* weight.item()* save_log(output[0][y])#-log(\pi)
				loss_list.append(loss) 
				action = y

		loss_total = sum(loss_list)/batch_size
		loss_total.backward()
		torch.nn.utils.clip_grad_norm(policyNet.parameters(), clip_norm)
		optimizer.step()
		optimizer.param_groups[0]['lr'] *= decay_rate
		
		if abs(pre_loss - loss_total.data.cpu()[0]) < 1e-4:
			break
		else:
			pre_loss = loss_total.data.cpu()[0]
			
		if num_step % 20==0:
			reward_test , loss_test= test_reward(policyNet, test_size)
			print("[Reset: %r; Online: %r; LR: %.3f; Epoch: %d; num_step: %d] Train loss: %.5f; Test loss: %.3f; Test reward: %.3f" %(adam_reset, online, lr, epoch, num_step, pre_loss, loss_test, reward_test))
		num_step +=1
	if num_step > MAX_STEP:
		break










