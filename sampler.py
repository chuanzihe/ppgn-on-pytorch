import os
import sys
import torch 
import torchvision
import torch.nn as nn
import numpy as np # replace all np with torch
from torch import optim
from model import EncPPGN
from model import GenPPGN
from model import CondPPGN
from torch.autograd import Variable
import scipy.misc
# import models.caffenet.caffenet as caffenet 
# from models.caffenet.caffenet import CaffeNet
import ipdb
# import genppgn

class Sampler():
	"""perform on class condition"""
	def __init__(self, config):

		# paths
		self.enc_path = config.enc_path
		self.gen_path = config.gen_path
		self.cond_path = config.cond_path
		self.h_dim = config.h_dim

		# sampling
		self.lr = config.samp_lr
		self.lr_end = config.samp_lr_end
		self.samp_iters = config.samp_iters
		self.caffenet = config.caffenet
		self.sample_path = config.sample_path

		# hyper parameters
		self.epsilon1 = config.epsilon1
		self.epsilon2 = config.epsilon2
		self.epsilon3 = config.epsilon3
		self.sample_shape = (3, 227, 227)

		self.softmax_exp = 1e-10
		self.clip_min = 0
		self.clip_max = 30
		self.ep1_low = 1e-6
		self.ep1_high = 1e-2

		self.init_code()
		self.load_pklmodel()

		self.xy = config.samp_xy
		self.unit = config.samp_unit

		self.reset_every = config.samp_reset
		self.save_every = config.samp_save


		# sampling values
		self.h = self.to_variable(torch.zeros([1,self.h_dim]))
		self.h_hat = self.to_variable(torch.zeros([1,self.h_dim]))
		self.x = self.to_variable(torch.zeros(self.sample_shape))
		self.c = self.to_variable(torch.zeros([1, 1000]))
		self.hook_grad()

		with open(config.synset_file, 'r') as synset_file:
			self.class_names = [line.split(",")[0].split(" ", 1)[1].rstrip('\n') for line in synset_file.readlines()]
		# under test
		# if config.caffenet:
		# 	self.load_caffemodel()
		# else:	
		# 	self.load_pklmodel()		

	def hook_grad(self):
		self.h.retain_grad()
		self.h_hat.retain_grad()
		self.x.retain_grad()
		self.c.retain_grad()

	def to_numpy(self):
		pass

	def save_image(self, img, filename):
 		torchvision.utils.save_image(self.denorm(img.data), 
 			os.path.join(self.sample_path, filename))
 		
 		# trial save images
 		# torchvision.utils.save_image(self.denorm(img.data),
 		# 	os.path.join(self.sample_path, filename),
 		# 	normalize=True)

	def normalize(self, img, out_range=(0.,1.), in_range=None):
		# save 2
		if not in_range:
			min_val = torch.min(img)
			max_val = torch.max(img)
		else:
			min_val = in_range[0]
			max_val = in_range[1]
	
		result = img.clone()
		result[result > max_val] = max_val
		result = (result - min_val) / (max_val - min_val) * (out_range[1] - out_range[0]) + out_range[0]
		return result

	def deprocess(self, images, out_range=(0.,1.), in_range=None):
		# save 2
		c   = images.size()[0]
		ih  = images.size()[1]
		iw  = images.size()[2]
	
		result = torch.zeros((ih, iw, c)) # h,w,c format
	
		# Normalize before saving
		result[:] = images.clone().permute(1,2,0)
		result = self.normalize(result, out_range, in_range)
		return result

	def save_image2(self, img, filename):
		img = img.data
		img = torch.from_numpy(img.cpu().numpy()[::-1,:,:].copy()).cuda() # Convert from BGR to RGB
		output_img = self.deprocess(img, in_range=(-120,120))
		filename = os.path.join(self.sample_path, filename)                
		scipy.misc.imsave(filename, output_img.numpy())

	def init_code(self):
		self.fixed_noise = self.to_variable(torch.randn(1, self.h_dim))

	def denorm(self, x):
		"""Convert range (-1, 1) to (0, 1)"""
		out = (x + 1) / 2
		return out.clamp(0, 1)

	def get_label(self, condition):
		unit = condition['unit']
		return self.class_names[unit]        

	def to_data(self, x):
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def to_variable(self, x, grad=True):
		if torch.cuda.is_available():
			x = x.cuda()
		return Variable(x, requires_grad=grad)        

	def load_mixedmodel(self):
		"""test only"""
		self.enc = CaffeNet()
		self.enc.load_state_dict(torch.load(self.enc_path))
		# bug: weight.0 not exist, forward method not implemented
		self.gen = GenPPGN()
		self.gen.load_state_dict(torch.load(self.gen_path))
		self.cond = CondPPGN()

		if torch.cuda.is_available():
			self.enc.cuda()
			self.gen.cuda()
			self.cond.cuda()

		print('model from mixed loaded')		

	def load_caffemodel(self):
		""" load from caffenet"""
		# self.cond = caffenet.caffenet
		self.enc = CaffeNet()
		self.enc.load_state_dict(torch.load(self.enc_path))
		self.gen = genppgn.genppgn
		self.gen.load_state_dict(torch.load(self.gen_path))
		print('model from caffe loaded')

	def load_pklmodel(self):
		""" load from pkl model"""
		self.enc = EncPPGN()
		self.cond = CondPPGN()
		self.gen = GenPPGN()
		self.gen.load_state_dict(torch.load(self.gen_path))
		if torch.cuda.is_available():
			self.enc.cuda()
			self.gen.cuda()
			self.cond.cuda()		
		print('model from pkl loaded')

	def grad_x_to_h(self, grad):
		""" backward from image x to h """
		# G'(h)
		# self.x generated at diff_h_DAE_h
		# @todo(chuanzi): backward on gen?
		# @todo(chuanzi): direct zero grad on data
		# grad size (1, 3, 227, 227)
		self.x.retain_grad()
		self.h.retain_grad()
		# seet h grad before and after
		self.x.backward(gradient=grad, retain_graph=True)
		g = self.h.grad.data.clone()

		self.x.grad.data.zero_()
		self.h.grad.data.zero_()

		return g

	def get_d_prior(self):
		""" DAE to approximate h to x to h' """	
		""" E(G(h)) - h """
		self.h.retain_grad()
		self.x = self.gen(self.h)
		self.x.retain_grad()
		# x_data = self.x.data.copy() # cropped in GEN as torch.Size([1, 3, 227, 227])
		if self.caffenet is False:
			self.h_hat, _ = self.enc(self.x) # return h, h1
		g = self.h_hat.data - self.h.data

		return g # torch.cuda.FloatTensor

	def get_d_condition(self, condition):

		""" get d_condition, G(h)to h and C(G(h) to h) """
		""" grad from cond to x -> grad from x to cond """

		unit = condition['unit']
		conv_xy = condition['xy'] # @todo(chuanzi): not used
		self.x.retain_grad()
		pred = self.cond(self.x) # (bs, 1, 1000), @todo(chuanzi): see shape
		pred.retain_grad()
		layer_acts = pred.data[0]
		one_hot = torch.zeros(pred.size())
		if torch.cuda.is_available():
			one_hot = one_hot.cuda()
		# layer_acts = self.c.data

		best_prob, best_unit = torch.max(layer_acts,0) #bug, not best prob, net no softmax

		# @todo(chuanzi): use torch.nn.functional.log_softmax
		exp_acts = torch.exp(layer_acts - torch.max(layer_acts))
		probs = exp_acts/(self.softmax_exp + torch.sum(exp_acts)) # no need to keepdim?

		softmax_grad = 1 - probs  #@diff: no copy for torch
		obj_prob = probs[unit]	#goal prob
		one_hot.view(-1)[unit] = softmax_grad[unit]
		pred.backward(gradient=one_hot, retain_graph=True)
		d_condition = self.x.grad.data
		self.cond.zero_grad()

		info = {
				'best_unit': best_unit.cpu()[0],
				'best_unit_prob': probs[best_unit].cpu()[0],
				}

		return d_condition, obj_prob, info

	# helper functions
	def save_samples(self,list_samples):

		for p in list_samples:
			image, name, label = p #lastxx, filename label
			torchvision.utils.save_image(self.denorm(image.data),
				os.path.join(self.sample_path, name))

	def print_progress(self, iter, info, condition, prob, d_condition):
		print('step: %04d\t max: %4s [%.2f]\t obj: %4s [%.2f]\t norm: [%.2f]' 
				% (iter, info['best_unit'], info['best_unit_prob'], 
					condition['unit'], prob, torch.norm(d_condition)))


	def sampling(self):

		last_img = torch.zeros(self.sample_shape)
		last_prop = -sys.maxsize # @todo(chuanzi): set as -1?

		#start code
		start_code = np.random.normal(0,1,(1,self.h_dim))
		start_code = (torch.from_numpy(start_code)).float() # @todo(chuanzi):  share same memory?
		print( ">>start code: [%.4f, %.4f]" % (torch.min(start_code), torch.max(start_code)))
		self.h = self.to_variable(start_code)
		condition_idx =  0
		list_samples = []
		iter = 0
		conditions = [{ "unit": int(u), "xy": self.xy } for u in self.unit.split('_')]   
		while True:

			step_size = self.lr + ((self.lr_end - self.lr) * iter)/ self.samp_iters
			condition = conditions[condition_idx]

			# ipdb.set_trace()
			# epsilon1, d log(p(h))/ dh modeled by DAE
			d_prior = self.get_d_prior() # gen self.x, self.h_hat inside

			# epsilon2, d cond to x by d x to h 

			d_condition_x, prob, info = self.get_d_condition(condition)
			d_condition = self.grad_x_to_h(d_condition_x)

			self.print_progress(iter, info, condition, prob, d_condition)

			# epsilon3
			if self.epsilon3 > 0:
				noise = np.random.normal(0, self.epsilon3, self.h.size())
				noise = torch.from_numpy(noise).float()
			else:
				noise = torch.zeros(self.h.size())

			if torch.cuda.is_available():
				noise = noise.cuda()

			# update dh

			# d_condition size not match
			d_h = self.epsilon1 * d_prior + self.epsilon2 * d_condition + noise

			# @todo(chuanzi): where can see gradient acsend
			# @todo(chuanzi): check kuohao place
			self.h.data += step_size/torch.abs(d_h).mean() * d_h

			self.h.data = torch.clamp(self.h.data, min=self.clip_min, max=self.clip_max)

			# @todo(chuanzi): : adjust structure

			last_img = self.x[0].clone()
			last_prob = prob

			# reset code

			if self.reset_every > 0 and iter % self.reset_every == 0 and iter > 0:
				start_code = np.random.normal(0, 1, self.h.size())
				start_code = (torch.from_numpy(start_code)).float()
				self.h = self.to_variable(start_code)

				self.epsilon1 = np.random.uniform(low=self.ep1_low, 
													high=self.ep1_high)

			if self.save_every > 0 and iter % self.save_every == 0:
				filename = "malasample-%s-%05d.jpg" % (condition['unit'], iter)
				label = self.get_label(condition)
				list_samples.append((last_img, filename, label))
				self.save_image(last_img, '1st%s'%(filename))
				self.save_image2(last_img, '2nd%s'%(filename))

			if torch.norm(d_h) == 0:
				print('d_h is 0')
				break

			if iter > 0 and iter % self.samp_iters == 0:
				condition_idx += 1
				if condition_idx == len(conditions):
					break

			iter += 1

		print('------------------')
		print('Last sample: prob [%s]' % last_prob)
		self.save_image(last_img, 'final_sample.png')
		self.save_image2(last_img, 'final_sample2nd.png')
		print ('saving final images')
		# self.save_samples(list_samples)