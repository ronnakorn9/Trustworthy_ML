# code is from https://github.com/meghdadk/SCRUB/blob/main/large_scale_unlearning.ipynb
import copy
import torch
import torch.nn as nn
import time
import numpy as np
from matplotlib import pyplot as plt
import os

from torch.utils.data import DataLoader


from repdistiller.KD import DistillKL
from repdistiller.util import adjust_learning_rate as sgda_adjust_learning_rate
from repdistiller.loops import train_distill, validate
from SCRUB_args import args
from SCRUB_util import avg_fn, get_optimizer, plot_performance 
from util import make_if_not_exist

def SCRUB_unlearn(model, forget_loader, retain_loader):

    teacher = copy.deepcopy(model)
    student = copy.deepcopy(model)

    model_t = copy.deepcopy(teacher)
    model_s = copy.deepcopy(student)


    swa_model = torch.optim.swa_utils.AveragedModel(
        model_s, avg_fn=avg_fn)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = get_optimizer(args.optim, trainable_list)

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        swa_model.cuda()

    acc_rs = []
    acc_fs = []
    acc_ts = []
    acc_vs = []
    for epoch in range(1, args.sgda_epochs + 1):

        lr = sgda_adjust_learning_rate(epoch, args, optimizer)

        print("==> SCRUB unlearning ...")

        acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls, args, True)
        acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, args, True)
        # acc_v, acc5_v, loss_v = validate(
        #     valid_loader_full, model_s, criterion_cls, args, True)
        acc_rs.append(100-acc_r.item())
        acc_fs.append(100-acc_f.item())
        # acc_vs.append(100-acc_v.item())

        maximize_loss = 0
        if epoch <= args.msteps:
            maximize_loss = train_distill(
                epoch, forget_loader, module_list, swa_model, criterion_list, optimizer, args, "maximize")
        train_acc, train_loss = train_distill(
            epoch, retain_loader, module_list, swa_model, criterion_list, optimizer, args, "minimize")
        if epoch >= args.sstart:
            swa_model.update_parameters(model_s)

        print("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(
            maximize_loss, train_loss, train_acc))
    acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls, args, True)
    acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, args, True)
    # acc_v, acc5_v, loss_v = validate(valid_loader_full, model_s, criterion_cls, args, True)
    acc_rs.append(100-acc_r.item())
    acc_fs.append(100-acc_f.item())
    # acc_vs.append(100-acc_v.item())

    plot_performance(acc_rs, acc_fs)

    return model_s

def scrub(model, forget_loader, retain_loader, reuse_scrub=False, name_append=""):
    scrub_model_location = "./output/checkpoints/"
    make_if_not_exist(scrub_model_location)
    if reuse_scrub:
        model_s = copy.deepcopy(model)
        selected_model = f"scrub_{args.model}_{args.dataset}_forgot{name_append}_seed{args.seed}_step{int(args.sgda_epochs)}.pt"
        selected_path = os.path.join(scrub_model_location, selected_model)
        model_s_final = copy.deepcopy(model_s)
        model_s.load_state_dict(torch.load(selected_path))
        return model_s, model_s_final

    
    model.train()
    # teacher = copy.deepcopy(model)
    # student = copy.deepcopy(model)

    # model_t = copy.deepcopy(teacher)
    # model_s = copy.deepcopy(student)
    model_t = copy.deepcopy(model)
    model_s = copy.deepcopy(model)

    #this is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    #For SGDA smoothing
    beta = 0.1
    def avg_fn(averaged_model_parameter, model_parameter, num_averaged): return (
        1 - beta) * averaged_model_parameter + beta * model_parameter
    swa_model = torch.optim.swa_utils.AveragedModel(
        model_s, avg_fn=avg_fn)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = get_optimizer(args.optim, trainable_list)
    
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        swa_model.cuda()


    t1 = time.time()
    acc_rs = []
    acc_fs = []
    # acc_vs = []
    # acc_fvs = []
    
    
    # forget_validation_loader = copy.deepcopy(valid_loader_full)
    # fgt_cls = list(np.unique(forget_loader.dataset.targets))
    # indices = [i in fgt_cls for i in forget_validation_loader.dataset.targets]
    # forget_validation_loader.dataset.data = forget_validation_loader.dataset.data[indices]
    # forget_validation_loader.dataset.targets = forget_validation_loader.dataset.targets[indices]
    
    total_time = 0.0
    scrub_name = f"scrub_{args.model}_{args.dataset}_forgot{name_append}_seed{args.seed}_step"
    for epoch in range(1, args.sgda_epochs + 1):

        lr = sgda_adjust_learning_rate(epoch, args, optimizer)

        acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls, args, True)
        acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, args, True)
        # acc_v, acc5_v, loss_v = validate(valid_loader_full, model_s, criterion_cls, args, True)
        # acc_fv, acc5_fv, loss_fv = validate(forget_validation_loader, model_s, criterion_cls, args, True)
        acc_rs.append(100-acc_r.item())
        acc_fs.append(100-acc_f.item())
        # acc_vs.append(100-acc_v.item())
        # acc_fvs.append(100-acc_fv.item())
        t1 = time.time()
        maximize_loss = 0
        if epoch <= args.msteps:
            maximize_loss = train_distill(epoch, forget_loader, module_list, swa_model, criterion_list, optimizer, args, "maximize", test_model = model)
        train_acc, train_loss = train_distill(epoch, retain_loader, module_list, swa_model, criterion_list, optimizer, args, "minimize",)
        if epoch >= args.sstart:
            swa_model.update_parameters(model_s)
        full_path = os.path.join(scrub_model_location, scrub_name+str(epoch)+".pt")
        torch.save(model_s.state_dict(), full_path)
        t2 = time.time()
        total_time += (t2 - t1)
        print ("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))
    
    print(f"[RESULT] SCRUB unlearn time: {total_time:.4f} seconds")


    acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls, args, True)
    acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, args, True)
    # acc_v, acc5_v, loss_v = validate(valid_loader_full, model_s, criterion_cls, args, True)
    # acc_fv, acc5_fv, loss_fv = validate(forget_validation_loader, model_s, criterion_cls, args, True)
    acc_rs.append(100-acc_r.item())
    acc_fs.append(100-acc_f.item())
    # acc_vs.append(100-acc_v.item())
    # acc_fvs.append(100-acc_fv.item())

    
    # indices = list(range(0,len(acc_rs)))
    # plt.plot(indices, acc_rs, marker='*', color=u'#1f77b4', alpha=1, label='retain-set')
    # plt.plot(indices, acc_fs, marker='o', color=u'#ff7f0e', alpha=1, label='forget-set')
    # ### plt.plot(indices, acc_vs, marker='^', color=u'#2ca02c',alpha=1, label='validation-set')
    # ### plt.plot(indices, acc_fvs, marker='.', color='red',alpha=1, label='forget-validation-set')
    # plt.legend(prop={'size': 14})
    # plt.tick_params(labelsize=12)
    # plt.xlabel('epoch',size=14)
    # plt.ylabel('error',size=14)
    # plt.grid()
    # plt.show()
    
    
    # try:
    #     selected_idx, _ = min(enumerate(acc_fs), key=lambda x: abs(x[1]-acc_fvs[-1]))
    # except:
    selected_idx = len(acc_fs) - 1

    print ("the selected index is {}".format(selected_idx))
    selected_model = "scrub_{}_{}_seed{}_step{}.pt".format(args.model, args.dataset, args.seed, int(selected_idx))
    selected_path = os.path.join(scrub_model_location, selected_model)
    model_s_final = copy.deepcopy(model_s)
    model_s.load_state_dict(torch.load(selected_path))
    
    
    return model_s, model_s_final

### self-implementation of SCRUB unlearn
def SCRUB(model, forget_data, retain_data):

    gamma = 1
    alpha = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_model = copy.deepcopy(model)
    student_model = copy.deepcopy(model)
    teacher_model.to(device)
    student_model.to(device)

    teacher_model.eval()
    student_model.train()

    criterion_retain = nn.KLDivLoss()
    criterion_obj = nn.CrossEntropyLoss()
    criterion_forget = nn.KLDivLoss()

    forget_loader = DataLoader(forget_data, batch_size=64)
    retain_loader = DataLoader(retain_data, batch_size=64)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    total_epoch = 5
    maxi_epoch = 2

    for epoch in range(total_epoch):
        if epoch < maxi_epoch:
            for f_input, f_target in forget_loader:
                f_input, f_target = f_input.to(device), f_target.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.no_grad():
                    teacher_outputs = teacher_model(f_input)
                student_outputs = student_model(f_input)

                # student_outputs = F.log_softmax(student_outputs, dim=1)

                # added -1 since we want to maximize the value
                f_loss = -1 * criterion_forget(student_outputs, teacher_outputs)

                # Backward pass and optimize
                f_loss.backward()
                optimizer.step()


        for r_input, r_target in retain_loader:
            r_input, r_target = r_input.to(device), r_target.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.no_grad():
                teacher_outputs_2 = teacher_model(r_input)
            student_outputs_2 = student_model(r_input)

            loss_div = criterion_retain(student_outputs_2, teacher_outputs_2)
            loss_cls = criterion_obj(student_outputs_2, r_target)

            loss = gamma * loss_cls + alpha * loss_div
            loss.backward()
            optimizer.step()

    return student_model