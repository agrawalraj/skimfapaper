
from math import floor
import time
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

from sklearn.model_selection import train_test_split

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

RELU_FN = nn.ReLU(inplace=False)


def dot(X, Z):
    return torch.mm(X, Z.t())


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
		M = optimization_config['M'] # Number of datapoints for CV loss
		truncScheduler = optimization_config['truncScheduler']
		param_save_freq = optimization_config['param_save_freq']
		valid_report_freq = optimization_config['valid_report_freq']
		lr = optimization_config['lr']
		train_noise = optimization_config['train_noise']
		noise_var_init = optimization_config['noise_var_init']

		# Initialize skimfa kernel hyperparams
		p = X_train.shape[1] # Total number of covariates
		Q = kernel_config['Q'] # Highest order interaction
		eta = torch.ones(Q, requires_grad=True) # Intialize global interaction variances to 1
		U_tilde = torch.ones(p, requires_grad=True) # Unconstrained parameter to generate kappa
		noise_var = torch.tensor(noise_var_init, requires_grad=train_noise)
		c = 0.

		if refit_after_selection:
			train_modes = ['selection', 'refit']

		else:
			train_modes = ['selection']

		all_training_losses = dict()
		all_validation_losses = dict()
		saved_params = dict()
		
		for train_mode in train_modes:
			
			all_training_losses[train_mode] = []
			all_validation_losses[train_mode] = []
			saved_params[train_mode] = []

			if train_mode == 'selection':
				T_gradient_steps = optimization_config['T']
				skimfa_kernel = kernel_config['skimfa_kernel']
				skimfa_kernel_fn = skimfa_kernel(kernel_config=kernel_config)
				truncScheduler = optimization_config['truncScheduler']

			else:
				# Refit eta and kappa after variable selection
				refit_after_selection = optimization_config['refit_after_selection']
				if refit_after_selection:
					saved_params['refit'] = dict()
					T_gradient_steps = optimization_config['T_refit']
					skimfa_kernel_refit = optimization_config['skimfa_kernel_refit']
					skimfa_kernel_fn_refit = skimfa_kernel_refit(kernel_config=kernel_config)
					truncScheduler = lambda t, U_tilde, c: 0

			# Gradient descent training loop
			for t in range(T_gradient_steps):
				X_cv_train, X_cv_test, Y_cv_train, Y_cv_test = train_test_split(X_train, Y_train, test_size=M)
				kappa = make_kappa(U_tilde, c)
				c = truncScheduler(t, U_tilde, c)
				K_train = skimfa_kernel_fn(X1=X_cv_train, X2=X_cv_train, kappa=kappa, eta=eta)
				K_test_train = skimfa_kernel_fn(X1=X_cv_test, X2=X_cv_train, kappa=kappa, eta=eta)
				alpha = kernel_ridge_weights(K_train, Y_cv_train, noise_var)
				L = cv_mse_loss(alpha, K_test_train, Y_cv_test)

				# Perform gradient decent step
				L.backward()
				kappa_unconstrained.data = kappa_unconstrained.data - lr * kappa_unconstrained.grad.data
				kappa_unconstrained.grad.zero_()

				if eta.requires_grad:
					eta.data = eta.data - lr * eta.grad.data
					eta.grad.zero_()

				if train_noise:
					noise_var.data = noise_var.data - lr * noise_var.grad.data
					noise_var.grad.zero_()

				if (t % valid_report_freq == 0) or t == (T_gradient_steps-1):
					K_valid_train = skimfa_kernel_fn(X1=X_valid, X2=X_cv_train, kappa=kappa, eta=eta)
					valid_mse = cv_mse_loss(alpha, K_valid_train, Y_valid)
					validation_losses.append(valid_mse)

				if (t % param_save_freq == 0) or t == (T_gradient_steps-1):
					saved_params[t] = dict()
					saved_params['U_tilde'] = U_tilde
					saved_params['eta'] = eta
					saved_params['noise_var'] = noise_var
					saved_params['c'] = c

			end_time = time.time()
			self.fitting_time_minutes = (end_time - start_time) / 60.


	def predict(self, X_test):
		pass


	def get_prod_measure_effect():
		pass


	def get_covariate_measure_effect():
		pass


class reFitSKIMFA(SKIMFA):
	pass





