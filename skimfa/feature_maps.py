
import torch
# from splines import *


def __feat_dims2cov_dims():
	pass


class FeatureMap(object):
	def __init__(self, covariate_dims, covariate_types):
		self.covariate_dims = covariate_dims
		self.covariate_types = covariate_types
	
	def make_feature_map(self, X_train):
		raise NotImplementedError

	def featmap(self, X):
		raise NotImplementedError

	def __call__(self, X):
		return self.featmap(X)


class LinearFeatureMap(FeatureMap):
	def make_feature_map(self, X_train):
		self.__dim_mean = X_train.mean(axis=0)
		self.__dim_sd = X_train.std(axis=0)
		self.input_dim_indcs = list(range(X_train.shape[1]))
		assert torch.sum(self.__dim_sd == 0) == 0, "Zero variance features"

	def featmap(self, X):
		return (X - self.__dim_mean) / self.__dim_sd


# ['auto-nonlinear', 'auto-linear']

# ['categorical', 'continuous', 'seasonal', 'ordinal']

# categorical --> same cov, do cateogy orth but don't diveide byy SD, linear
# ordinal ---> mean SD scale, linear
# covariate extrapolate

# 'continuous' ---> bspline quantile knots
# 'seasonal' ---> wavelet basis
#  choose number of knots equal to 

if __name__ == "__main__":
	p = 10
	X_train = torch.normal(mean=0., std=1., size=(1000, p))
	X_test = torch.normal(mean=0., std=1., size=(100, p))

	covariate_dims = list(range(p))
	covariate_types = ['continuous'] * p
	print(covariate_types)

	linfeatmap = LinearFeatureMap(covariate_dims, covariate_types)
	linfeatmap.make_feature_map(X_train)
	print(linfeatmap(X_test))	

