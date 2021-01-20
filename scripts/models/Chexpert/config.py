import warnings
import torch

class DefaultConfig(object):

	bs = 32
	use_gpu = True
	num_workers = 2
	
	max_epoch = 20
	lr = 0.0001
	betas = (0.9, 0.999)
	eps = 1e-08
	lr_decay = 0.95
	weight_decay = 1e-5
	device = torch.device('cuda:0')

opt = DefaultConfig()
