
import torch 
from feature_maps import *


# Optimizing higher-order interaction kernels using coordinate descent, O(1) update to kernel

def dot(X, Z):
    return torch.mm(X, Z.t())


def basis_expansion_pairwise_skimfa_kernel_uncorrected(X, Z, eta1, eta2, c, rescale):
    eta1sq, eta2sq = eta1.pow(2.0), eta2.pow(2.0)
    k1 = 0.5 * eta2sq * (1.0 + dot(X, Z)).pow(2.0)
    k2 = -0.5 * eta2sq * dot(X.pow(2.0), Z.pow(2.0))
    k3 = (eta1sq - eta2sq) * dot(X, Z)
    k4 = c ** 2 - 0.5 * eta2sq
    return (k1 + k2 + k3 + k4) * rescale


def basis_expansion_pairwise_skimfa_kernel_corrected(X, Z, eta2, kappa, input_dim_indcs, rescale):
	K_uncorrected = basis_expansion_pairwise_skimfa_kernel_uncorrected(X, Z, eta, rescale)
	var_block_dict = dict()
	var_indcs = set(input_dim_indcs)
	
	for var_ix in var_indcs:
		var_block_dict[var_ix] = []
	for feat_ix_pos, var_ix in enumerate(input_dim_indcs):
		var_block_dict[var_ix].append(feat_ix_pos)
	
	N1 = X.shape[0]
	N2 = Z.shape[0]
	K_correction = torch.zeros((N1, N2))
	
	for var_ix in var_indcs:
		dim_indcs = var_block_dict[var_ix]
		K_correction += kappa[var_ix].pow(4) * kernel(X[:, dim_indcs], Z[:, dim_indcs], 
												torch.tensor(0.), eta2, torch.tensor(0.), rescale=1.)
	
	return K_uncorrected - K_correction * rescale


class SKIMFAKernel(object):
	def __init__(self, train_valid_data, kernel_config):
		self.kernel_config = kernel_config
		self.feat_map = kernel_config['feat_map']
		self.input_dim_indcs = self.feat_map.input_dim_indcs.copy()

		if kernel_config['cache']:
			X_train = train_valid_data['X_train']
			X_valid = train_valid_data['X_valid']
			self.train_valid_data = train_valid_data
			self.X_feat_train = self.feat_map(X_train)
			self.X_feat_valid = self.feat_map(X_valid)

	def kernel_matrix(self, X1, X2, kappa, eta, X1_info=None, X2_info=None):
		raise NotImplementedError


class PairwiseSKIMFABasisKernel(SKIMFAKernel):
	def kernel_matrix(self, X1, X2, kappa, eta, X1_info=None, X2_info=None):
		if (X1_info == None) or (self.kernel_config['cache'] == False):
			X1_feat = self.feat_map(X1)
		else:
			X1_feat = self.X_feat_train[X1_info, :]

		if (X2_info == None) or (self.kernel_config['cache'] == False):
			X2_feat = self.feat_map(X2)
		else:
			X2_feat = self.X_feat_train[X2_info, :]
		
		kappa_expanded = kappa[self.input_dim_indcs]
		rescale = self.kernel_config['rescale']

		if self.kernel_config['uncorrected']:
			return basis_expansion_pairwise_skimfa_kernel_uncorrected(kappa_expanded * X1_feat, kappa_expanded * X2_feat, eta[1], eta[2], eta[0], rescale)

		else:
			return basis_expansion_pairwise_skimfa_kernel_corrected(kappa_expanded * X1_feat, kappa_expanded * X2_feat, eta[2], kappa, 
																	  self.input_dim_indcs, rescale)


class DistributedSKIMFAKernel(SKIMFAKernel):
	def __init__(self, kernel_config, system_config):
		pass


if __name__ == "__main__":
	# Make feature map 
	p = 10
	X_train = torch.normal(mean=0., std=1., size=(1000, p))
	X_test = torch.normal(mean=0., std=1., size=(100, p))
	X_valid = torch.normal(mean=0., std=1., size=(100, p))
	train_valid_data = dict()
	train_valid_data['X_train'] = X_train
	train_valid_data['X_valid'] = X_valid


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

	pair_skim = PairwiseSKIMFABasisKernel(train_valid_data, kernel_config)

	# Compute kernel matrix
	kappa = torch.rand(p)
	eta = torch.tensor([2., 2., 2.])

	print(pair_skim.kernel_matrix(X_train, X_train, kappa, eta))




