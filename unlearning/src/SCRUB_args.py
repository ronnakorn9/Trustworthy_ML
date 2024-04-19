### separate config file for SCRUB unlearning

from  argparse import Namespace
args = Namespace()
args.model = "resnet50"
args.dataset = "cifar10"
args.seed = 42

args.optim = 'sgd'
args.gamma = 1
args.alpha = 0.5
args.beta = 0
args.smoothing = 0.5
args.msteps = 3
args.clip = 0.2
args.sstart = 10
args.kd_T = 4
args.distill = 'kd'

args.sgda_batch_size = 128
args.del_batch_size = 32
args.sgda_epochs = 3
args.sgda_epochs = 5
args.sgda_learning_rate = 0.0005
args.lr_decay_epochs = [3,5,9]
args.lr_decay_rate = 0.1
args.sgda_weight_decay = 5e-4
args.sgda_momentum = 0.9