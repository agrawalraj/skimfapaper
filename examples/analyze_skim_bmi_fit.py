
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


skimfit = pickle.load(open('skimfit_bmi_snp.pkl', 'rb'))
bmi_snp_fit_data = pickle.load(open('bmi_snp_fit_data.pkl', 'rb'))

data = bmi_snp_fit_data['data']
X_valid = bmi_snp_fit_data['X_valid']
Y_valid = bmi_snp_fit_data['Y_valid']

var_names = np.array(list(data.drop(['BMI', 'sample_id'], axis=1).columns))

all_effects = dict()

for var_ix in skimfit.get_selected_covariates():
	all_effects[var_names[var_ix]] = get_linear_iteraction_effect(skimfit, [var_ix.item()])


for var_ix1 in skimfit.get_selected_covariates():
	for var_ix2 in skimfit.get_selected_covariates():
		if var_ix1 < var_ix2:
			all_effects[(var_names[var_ix1], var_names[var_ix2])] = get_linear_iteraction_effect(skimfit, [var_ix1.item(), var_ix2.item()])


print(f'Var(Y_valid): {round(Y_valid.var().item(), 1)}')
print(f'Mean-Squared Prediction Error on Test: {round(torch.mean((Y_valid -  skimfit.predict(X_valid)) ** 2).item(), 1)}')

print(var_names[skimfit.get_selected_covariates()])

for effect_name in all_effects:
	if abs(all_effects[effect_name]) >= .1:
		print(effect_name, all_effects[effect_name])

# Get top 5 main effects

