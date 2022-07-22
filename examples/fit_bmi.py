
import sys
import pickle
sys.path.append('../')
sys.path.append('../skimfa')  # TODO: replace once code is a python package

import math
import pandas as pd
import numpy as np
from skimfa.kernels import PairwiseSKIMFABasisKernel
from feature_maps import LinearFeatureMap
from fit import *
from sklearn.model_selection import train_test_split


def adaptive_cutoff_scheduler(t, U_tilde, prev_cutoff, r=.001):
	if t < 500:
		return 0
	if t == 500:
		c = get_percentile_thresh(U_tilde, percentile=.25)
		return c
	return max(min((1+r)*prev_cutoff, .75), prev_cutoff)


# Set seed for reproducibility
seed = 4321

torch.manual_seed(seed)
data = pd.read_pickle('bmi_snp_data.pkl')

Y = data['BMI'].astype(np.float64).values.copy()
avg_Y = Y.mean()

# Convert to torch tensors / center response and covariates
Y = torch.FloatTensor((Y - avg_Y)) # Demean response
X = data.drop(['BMI', 'sample_id'], axis=1).values.copy()
X = (X - X.mean(axis=0)) / X.std(axis=0)
X = torch.FloatTensor(X)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=.15, random_state=seed)

# Set hyperparams
p = X_train.shape[0]
covariate_dims = list(range(p))
covariate_types = ['continuous'] * p # irrelevant for now (in the future the selected feature map will depend on the covariate type)
linfeatmap = LinearFeatureMap(covariate_dims, covariate_types)
linfeatmap.make_feature_map(X_train) 

# Step 2: Make kernel configuration
kernel_config = dict()
kernel_config['uncorrected'] = True
kernel_config['rescale'] = 1.
kernel_config['feat_map'] = linfeatmap
kernel_config['cache'] = True
kernel_config['Q'] = 2 # all main and pairwise interaction effects

# Step 3: Make optimization configuration
optimization_config = dict()
optimization_config['T'] = 5000 # 2000 total gradient steps
optimization_config['M'] = 25 # size of cross-validation random sample
optimization_config['param_save_freq'] = 100 # save model weights every 100 iterations
optimization_config['valid_report_freq'] = 100 # how often to report MSE on validation set 
optimization_config['lr'] = .01
optimization_config['train_noise'] = False
optimization_config['noise_var_init'] = .5 * Y_train.var().detach().item()
optimization_config['truncScheduler'] = adaptive_cutoff_scheduler

# Fit SKIM-FA
train_valid_data = dict()
train_valid_data['X_train'] = X_train
train_valid_data['Y_train'] = Y_train
train_valid_data['X_valid'] = X_valid
train_valid_data['Y_valid'] = Y_valid

# VERY STRANGE error on my computer where I need to invert a matrix to not get a segmentation 11 fault error...
X_weird = np.random.normal(size=(500, 100))
np.linalg.inv(X_weird.T.dot(X_weird))

# Fit model
skimfit = SKIMFA()
skimfit.fit(train_valid_data, PairwiseSKIMFABasisKernel, kernel_config, optimization_config)

# Save fit
pickle.dump(skimfit, open('skimfit_bmi_snp.pkl', 'wb'))
pickle.dump({'data': data, 'X_train': X_train, 'X_valid': X_valid, 'Y_train': Y_train, 'Y_valid': Y_valid, 'avg_Y': avg_Y}, open('bmi_snp_fit_data.pkl', 'wb'))

