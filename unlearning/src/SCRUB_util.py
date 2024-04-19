### utility function that assist/work with SCRUN unlearning

import copy
import torch
from torch.utils.data import DataLoader
from SCRUB_args import args
from matplotlib import pyplot as plt

# this is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
# For SGDA smoothing
def avg_fn(averaged_model_parameter, model_parameter, num_averaged, beta=0.1): return (
    1 - beta) * averaged_model_parameter + beta * model_parameter


def get_optimizer(optim, trainable_list):
    if optim == "sgd":
        optimizer = torch.optim.SGD(trainable_list.parameters(),
                                    lr=args.sgda_learning_rate,
                                    momentum=args.sgda_momentum,
                                    weight_decay=args.sgda_weight_decay)
    elif optim == "adam":
        optimizer = torch.optim.Adam(trainable_list.parameters(),
                                    lr=args.sgda_learning_rate,
                                    weight_decay=args.sgda_weight_decay)
    elif optim == "rmsp":
        optimizer = torch.optim.RMSprop(trainable_list.parameters(),
                                        lr=args.sgda_learning_rate,
                                        momentum=args.sgda_momentum,
                                        weight_decay=args.sgda_weight_decay)
    return optimizer

def plot_performance(acc_rs, acc_fs):
    
    indices = list(range(0,len(acc_rs)))
    plt.plot(indices, acc_rs, marker='*', alpha=1, label='retain-set')
    plt.plot(indices, acc_fs, marker='o', alpha=1, label='forget-set')
    # plt.plot(indices, acc_vs, marker='.', alpha=1, label='valid-set')
    plt.legend(prop={'size': 14})
    plt.tick_params(labelsize=12)
    plt.title('SCRUB retain-, valid- and forget- set error',size=18)
    plt.xlabel('epoch',size=14)
    plt.ylabel('error',size=14)
    plt.show()

def get_dataloaders(train_dataset_full, num_to_forget, class_to_forget=None):
    if class_to_forget is None:
        ### random forget on any class
        forget_dataset, retain_dataset = torch.utils.data.random_split(train_dataset_full, [num_to_forget, len(train_dataset_full) - num_to_forget])

    # TODO: add more unlearn mode
    assert(len(forget_dataset) + len(retain_dataset) == len(train_dataset_full))

    forget_loader = DataLoader(forget_dataset, batch_size=64, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=64, shuffle=True)


    return (forget_loader, retain_loader), (forget_dataset, retain_dataset)