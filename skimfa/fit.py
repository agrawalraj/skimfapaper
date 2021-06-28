
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

	def fit(self, train_valid_data, skimfa_kernel_fn, kernel_config, optimization_config):
		self.kernel_config = kernel_config
		self.optimization_config = optimization_config

		start_time = time.time()

		# Load in training and validation data
		X_train = train_valid_data['X_train']
		Y_train = train_valid_data['Y_train']
		X_valid = train_valid_data['X_valid']
		Y_valid = train_valid_data['Y_valid']

		# Initialize skimfa kernel hyperparams
		p = X_train.shape[1] # Total number of covariates
		Q = kernel_config['Q'] # Highest order interaction
		eta = torch.ones(Q, requires_grad=True) # Intialize global interaction variances to 1
		U_tilde = torch.ones(p, requires_grad=True) # Unconstrained parameter to generate kappa
		noise_var = torch.tensor(, requires_grad=)
		c = 0.

		# Load in optimization specs
		T = optimization_config['T'] # Number of gradient descent steps
		M = optimization_config['M'] # Number of datapoints for CV loss
		truncScheduler = optimization_config['truncScheduler']
		param_save_freq = optimization_config['param_save_freq']
		valid_report_freq = optimization_config['valid_report_freq']
		lr = optimization_config['lr']

		# Gradient descent training loop
		training_losses = []
		validation_losses = []
		saved_params = dict()
		for t in T:
			X_cv_train, X_cv_test, Y_cv_train, Y_cv_test = train_test_split(X_train, Y_train, test_size=M)
			kappa = make_kappa(U_tilde, c)
			c = truncScheduler(t, U_tilde, c)
			K_train = skimfa_kernel_fn(X1=X_cv_train, X2=X_cv_train, kappa=kappa, eta=eta, kernel_config=kernel_config)
			K_test_train = skimfa_kernel_fn(X1=X_cv_test, X2=X_cv_train, kappa=kappa, eta=eta, kernel_config=kernel_config)
			alpha = kernel_ridge_weights(K_train, Y_cv_train, noise_var)
			L = cv_mse_loss(alpha, K_test_train, Y_cv_test)

			# Perform gradient decent step
			L.backward()
			kappa_unconstrained.data = kappa_unconstrained.data - lr * kappa_unconstrained.grad.data
			kappa_unconstrained.grad.zero_()

			if eta.requires_grad:
				eta.data = eta.data - lr * eta.grad.data
				eta.grad.zero_()

			TRAIN NOISE???

			if t % valid_report_freq == 0:
				K_valid_train = skimfa_kernel_fn(X1=X_valid, X2=X_cv_train, kappa=kappa, eta=eta, kernel_config=kernel_config)
				valid_mse = cv_mse_loss(alpha, K_valid_train, Y_valid)
				validation_losses.append(valid_mse)

			if t % param_save_freq == 0:
				saved_params[t] = dict()
				saved_params['U_tilde'] = U_tilde
				saved_params['eta'] = eta
				saved_params['noise_var'] = noise_var
				saved_params['c'] = c

		end_time = time.time()
		self.fitting_time_minutes = (end_time - start_time) / 60.


