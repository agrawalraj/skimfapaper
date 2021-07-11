
from math import floor
import time
import os

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

RELU_FN = nn.ReLU(inplace=False)


def make_kappa(U_tilde, c):
	return 1 / (1 - c) * RELU_FN((U_tilde ** 2 / (1 + U_tilde ** 2)) - c)


def kernel_ridge_weights(K, Y, noise_var):
	return (K + noise_var*torch.eye(K.shape[0])).inverse().mv(Y)


def cv_mse_loss(alpha, K_test_train, y_test):
	return torch.mean((y_test - K_test_train.mv(alpha)) ** 2)


def get_percentile_thresh(kappa_unconstrained, percentile=.25):
	kappa_normalized_sorted = torch.sort(make_fanova_local_scale(kappa_unconstrained, cutoff=0))[0]
	pos = floor(percentile * kappa_normalized_sorted.shape[0])
	return kappa_normalized_sorted[pos].item()


class SKIMFA(object):
	def __init__(self, gpu=False):
		assert gpu == False, 'Not Implemented GPU support yet'

	def fit(self, train_valid_data, kernel_config, optimization_config):
		self.kernel_config = kernel_config
		self.optimization_config = optimization_config

		start_time = time.time()

		# Load in training and validation data
		X_train = train_valid_data['X_train']
		Y_train = train_valid_data['Y_train']
		X_valid = train_valid_data['X_valid']
		Y_valid = train_valid_data['Y_valid']

		# Load in optimization specs
		N = X_train.shape[0]
		T = optimization_config['T']
		M = optimization_config['M'] # Number of datapoints for CV loss
		truncScheduler = optimization_config['truncScheduler']
		param_save_freq = optimization_config['param_save_freq']
		valid_report_freq = optimization_config['valid_report_freq']
		lr = optimization_config['lr']
		train_noise = optimization_config['train_noise']
		noise_var_init = optimization_config['noise_var_init']
		truncScheduler = optimization_config['truncScheduler']

		# Initialize skimfa kernel hyperparams
		p = X_train.shape[1] # Total number of covariates
		Q = kernel_config['Q'] # Highest order interaction
		eta = torch.ones(Q + 1, requires_grad=True) # Intialize global interaction variances to 1
		U_tilde = torch.ones(p, requires_grad=True) # Unconstrained parameter to generate kappa
		noise_var = torch.tensor(noise_var_init, requires_grad=train_noise)
		c = 0.
		skimfa_kernel = kernel_config['skimfa_kernel']
		skimfa_kernel_fn = skimfa_kernel(train_valid_data=train_valid_data, kernel_config=kernel_config)

		training_losses = []
		validation_losses = []
		saved_params = dict()
	
		# Gradient descent training loop
		for t in range(T):
			random_indcs = torch.randperm(N)
			cv_train_indcs = random_indcs[M:]
			cv_test_indcs = random_indcs[:M]
	
			X_cv_train = X_train[cv_train_indcs, :]
			X_cv_test = X_train[cv_test_indcs, :]
			Y_cv_train = Y_train[cv_train_indcs]
			Y_cv_test = Y_train[cv_test_indcs]

			kappa = make_kappa(U_tilde, c)
			c = truncScheduler(t, U_tilde, c)
			K_train = skimfa_kernel_fn.kernel_matrix(X1=X_cv_train, X2=X_cv_train, kappa=kappa, eta=eta, 
									   X1_info=cv_train_indcs, 
									   X2_info=cv_train_indcs)

			K_test_train = skimfa_kernel_fn.kernel_matrix(X1=X_cv_test, X2=X_cv_train, kappa=kappa, eta=eta,
										    X1_info=cv_test_indcs, X2_info=cv_train_indcs)

			alpha = kernel_ridge_weights(K_train, Y_cv_train, noise_var)
			L = cv_mse_loss(alpha, K_test_train, Y_cv_test)

			# Perform gradient decent step
			L.backward()
			U_tilde.data = U_tilde.data - lr * U_tilde.grad.data
			U_tilde.grad.zero_()

			if eta.requires_grad:
				eta.data = eta.data - lr * eta.grad.data
				eta.grad.zero_()

			if train_noise:
				noise_var.data = noise_var.data - lr * noise_var.grad.data
				noise_var.grad.zero_()

			if (t % valid_report_freq == 0) or (t == (T-1)):
				K_valid_train = skimfa_kernel_fn.kernel_matrix(X1=X_valid, X2=X_cv_train, kappa=kappa, eta=eta)

				valid_mse = cv_mse_loss(alpha, K_valid_train, Y_valid)
				validation_losses.append(valid_mse)

			if (t % param_save_freq == 0) or (t == (T-1)):
				saved_params[t] = dict()
				saved_params['U_tilde'] = U_tilde
				saved_params['eta'] = eta
				saved_params['noise_var'] = noise_var
				saved_params['c'] = c

		end_time = time.time()
		self.fitting_time_minutes = (end_time - start_time) / 60.


	def predict(self, X_test):
		pass


	def get_selected_covariates(self, X_test, iter=None):
		pass


	def get_prod_measure_effect(iter=None):
		pass


	def get_covariate_measure_effect(iter=None):
		pass
		# assert self.Q <= 2
		# if self.Q == 1:
		# 	return self.get_prod_measure_effect()

		# else: 


class reFitSKIMFA(SKIMFA):
	pass

	# refit_kernel_config, refit opt config


class reFitSKIMFACV(reFitSKIMFA):
	pass


if __name__ == "__main__":
	from kernels import *
	from feature_maps import *

	# Make feature map 
	p = 10
	X_train = torch.normal(mean=0., std=1., size=(1000, p))
	X_test = torch.normal(mean=0., std=1., size=(100, p))
	X_valid = torch.normal(mean=0., std=1., size=(100, p))

	X_train_np = np.random.normal(size=(500, 100))
	np.linalg.inv(X_train_np.T.dot(X_train_np))
	
	theta_true = torch.normal(mean=2., std=1., size=(p,))
	Y_train = X_train.mv(theta_true) + torch.normal(mean=0., std=1., size=(1000, ))
	Y_test = X_test.mv(theta_true) + torch.normal(mean=0., std=1., size=(100, ))
	Y_valid = X_valid.mv(theta_true) + torch.normal(mean=0., std=1., size=(100, ))

	train_valid_data = dict()
	train_valid_data['X_train'] = X_train
	train_valid_data['Y_train'] = Y_train
	train_valid_data['X_valid'] = X_valid
	train_valid_data['Y_valid'] = Y_valid

	covariate_dims = list(range(p))
	covariate_types = ['continuous'] * p

	linfeatmap = LinearFeatureMap(covariate_dims, covariate_types)
	linfeatmap.make_feature_map(X_train)

	# Make kernel config
	kernel_config = dict()
	kernel_config['uncorrected'] = True
	kernel_config['rescale'] = 1.
	kernel_config['feat_map'] = linfeatmap
	kernel_config['cache'] = True
	kernel_config['Q'] = 2
	kernel_config['skimfa_kernel'] = PairwiseSKIMFABasisKernel

	# Make optimization config
	optimization_config = dict()
	optimization_config['T'] = 2000
	optimization_config['M'] = 100
	optimization_config['param_save_freq'] = 100
	optimization_config['valid_report_freq'] = 100
	optimization_config['lr'] = .1
	optimization_config['train_noise'] = False
	optimization_config['noise_var_init'] = Y_train.var().detach().item()
	optimization_config['truncScheduler'] = lambda t, U_tilde, c: 0

	skimfit = SKIMFA()
	skimfit.fit(train_valid_data, kernel_config, optimization_config)

