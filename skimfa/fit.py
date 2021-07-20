
from math import floor
from tqdm import tqdm
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


def get_percentile_thresh(U_tilde, percentile=.25):
	kappa_normalized_sorted = torch.sort(make_kappa(U_tilde, c=0))[0]
	pos = floor(percentile * kappa_normalized_sorted.shape[0])
	return kappa_normalized_sorted[pos].item()


def adaptive_cutoff_scheduler(t, U_tilde, prev_cutoff, r=.01):
	if t < 500:
		return 0
	if t == 500:
		# remove 25% of the covariates
		c = get_percentile_thresh(U_tilde, percentile=.25)
		return c
	return max(min((1+r)*prev_cutoff, .75), prev_cutoff)


class SKIMFA(object):
	def __init__(self, gpu=False):
		assert gpu == False, 'Not Implemented GPU support yet'

	def fit(self, train_valid_data, skimfa_kernel, kernel_config, optimization_config):
		self.kernel_config = kernel_config
		self.skimfa_kernel = skimfa_kernel
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
		skimfa_kernel_fn = skimfa_kernel(train_valid_data=train_valid_data, kernel_config=kernel_config)

		training_losses = []
		validation_losses = []
		saved_params = dict()
	
		# Gradient descent training loop
		for t in tqdm(range(T)):
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
				print(f'Mean-Squared Prediction Error on Validation (Iteration={t}): {round(valid_mse.item(), 3)}')
				print(f'Number Covariates Selected={torch.sum(kappa > 0).item()}')

			if (t % param_save_freq == 0) or (t == (T-1)):
				saved_params[t] = dict()
				saved_params['U_tilde'] = U_tilde
				saved_params['eta'] = eta
				saved_params['noise_var'] = noise_var
				saved_params['c'] = c

		end_time = time.time()
		self.fitting_time_minutes = (end_time - start_time) / 60.

		# Store final fitted parameters
		last_iter_params = dict()
		last_iter_params['kappa'] = make_kappa(U_tilde, c)
		last_iter_params['eta'] = eta
		last_iter_params['noise_var'] = noise_var

		# Get kernel ridge weights using full training set
		K_train = skimfa_kernel_fn.kernel_matrix(X1=X_train, X2=X_train, kappa=kappa, eta=eta, 
									   X1_info=torch.arange(N), 
									   X2_info=torch.arange(N))

		alpha = kernel_ridge_weights(K_train, Y_train, noise_var)
		last_iter_params['alpha'] = alpha
		last_iter_params['X_train'] = X_train
		self.last_iter_params = last_iter_params

		# Store selected covariates
		self.selected_covariates = torch.where(self.last_iter_params['kappa'] > 0)[0]

	def predict(self, X_test):
		kappa = self.last_iter_params['kappa']
		eta = self.last_iter_params['eta']
		alpha = self.last_iter_params['alpha']
		X_train = self.last_iter_params['X_train']
		skimfa_kernel = self.skimfa_kernel
		kernel_config = self.kernel_config.copy()
		kernel_config['cache'] = False
		skimfa_kernel_fn = skimfa_kernel(train_valid_data=None, kernel_config=kernel_config)
		K_test_train = skimfa_kernel_fn.kernel_matrix(X1=X_test, X2=X_train, kappa=kappa, eta=eta)
		return K_test_train.mv(alpha)

	def get_selected_covariates(self):
		return self.selected_covariates.clone()

	def get_prod_measure_effect(self, X, V, iter=None):
		kappa = self.last_iter_params['kappa']
		iter_order = len(V)
		eta_V = self.last_iter_params['eta'][iter_order]
		theta_V = (eta_V * torch.prod(kappa[V])) ** 2
		if theta_V == 0:
			return torch.zeros(X.shape[0])
		else:
			skimfa_kernel = self.skimfa_kernel
			kernel_config = self.kernel_config.copy()
			kernel_config['cache'] = False
			skimfa_kernel_fn = skimfa_kernel(train_valid_data=None, kernel_config=kernel_config)
			K_V = 1.
			alpha = self.last_iter_params['alpha']
			X_train = self.last_iter_params['X_train']
			for cov_ix in V:
				K_V *= skimfa_kernel_fn.zero_mean_one_dim_kernel_matrix(X, X_train, cov_ix)
			
			return theta_V * K_V.mv(alpha)

	def get_covariate_measure_effect(self, iter=None):
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


def get_linear_iteraction_effect(skimfamodel, V):
	p = skimfamodel.last_iter_params['X_train'].shape[1]
	var_selected = set(skimfamodel.selected_covariates.detach().numpy())

	if len(set(V) - var_selected) > 0: # If there exists a covariate in V not selected, fV effect = 0
		return 0

	else:
		if len(V) == 0:
			raise NotImplementedError

		elif len(V) == 1:
			cov_ix = V[0]
			X_test = torch.zeros((2, p))
			X_test[1, cov_ix] = 1 
			est_effect = skimfamodel.get_prod_measure_effect(X_test, V)
			return (est_effect[1] - est_effect[0]).item()

		elif len(V) == 2:
			cov_ix1, cov_ix2 = V
			X_test = torch.zeros((2, p))
			X_test[:, cov_ix1] = 1
			X_test[1, cov_ix2] = 1
			est_effect = skimfamodel.get_prod_measure_effect(X_test, V)
			return (est_effect[1] - est_effect[0]).item()

		else:
			raise NotImplementedError


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

	# Make optimization config
	optimization_config = dict()
	optimization_config['T'] = 200
	optimization_config['M'] = 100
	optimization_config['param_save_freq'] = 100
	optimization_config['valid_report_freq'] = 100
	optimization_config['lr'] = .1
	optimization_config['train_noise'] = False
	optimization_config['noise_var_init'] = Y_train.var().detach().item()
	optimization_config['truncScheduler'] = lambda t, U_tilde, c: 0

	skimfit = SKIMFA()
	skimfit.fit(train_valid_data, PairwiseSKIMFABasisKernel, kernel_config, optimization_config)

	print(skimfit.get_selected_covariates())

	print(torch.mean((Y_test - skimfit.predict(X_test)) ** 2))

	print(torch.mean(skimfit.get_prod_measure_effect(X_test, [1, 2]) ** 2))

	true_effect = X_test[:, 1] * theta_true[1]
	fitted_effect = skimfit.get_prod_measure_effect(X_test, [1])

	print(torch.mean((true_effect - fitted_effect) ** 2))
	print(torch.mean(true_effect ** 2))

