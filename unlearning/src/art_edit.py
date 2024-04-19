# discliamer: most of this code is generated ussing chatGPT
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

from train import train_model, save_model, load_model
from util import test, cm_score
from model import get_model

# MIA attacker model
def init_weights(m):
    if isinstance(m, nn.Linear): 
        torch.nn.init.xavier_uniform_(m.weight)

class MembershipInferenceAttackModel(nn.Module):
    def __init__(self, num_classes, num_features=None):
        self.num_classes = num_classes
        if num_features:
            self.num_features = num_features
        else:
            self.num_features = num_classes

        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),

            # nn.Linear(64, 1),
            # nn.ReLU(),
        )
        self.labels = nn.Sequential(
            nn.Linear(num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(64 * 2, 1),
        )
        self.output = nn.Sigmoid()

        self.features.apply(init_weights)
        self.labels.apply(init_weights)
        self.combine.apply(init_weights)

    

    def forward(self, x_1, label):
        out_x1 = self.features(x_1)
        out_l = self.labels(label)
        is_member = self.combine(torch.cat((out_x1, out_l), 1))
        return self.output(is_member)
    # def forward(self, x_1, label):
    #     out_x1 = self.features(x_1)
    #     return self.output(out_x1)


def make_if_not_exist(directory):
    """
    given a directory address, create directory a new directory if it's not already exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_shadow_dataset(
    dataset,
    shadow_models,
    device,
    member_ratio: float = 0.5,
    num_epochs=10,
    disjoint_datasets=False,
    reload=False

) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Generates a shadow dataset (member and nonmember samples and their corresponding model predictions) by splitting
    the dataset into training and testing samples, and then training the shadow models on the result.

    :param dataset: The dataset used to train the shadow models.
    :param shadow_models: List of shadow models.
    :param member_ratio: Percentage of the data that should be used to train the shadow models. Must be between 0
                         and 1.
    :return: The shadow dataset generated. The shape is `((member_samples, true_label, model_prediction),
             (nonmember_samples, true_label, model_prediction))`.
    """

    # all shadow model receive the same dataset (non-disjoint dataset)
    if disjoint_datasets:
        shadow_dataset_size = len(dataset) // len(shadow_models)
    else:
        shadow_dataset_size = len(dataset)

    member_samples = []
    member_true_label = []
    member_prediction = []
    nonmember_samples = []
    nonmember_true_label = []
    nonmember_prediction = []

    # Train and create predictions for every model
    for i, shadow_model in enumerate(shadow_models):
        make_if_not_exist("./model/shadow_2/")
        model_filename = f"./model/shadow_2/shadow_{i}.pt"

        shadow_model.to(device)
        if disjoint_datasets:
            indices = list(range(shadow_dataset_size * i,
                           shadow_dataset_size * (i + 1)))
        else:
            indices = list(range(shadow_dataset_size))

        np.random.shuffle(indices)
        train_size = int(len(indices) * member_ratio)
        train_sampler = SubsetRandomSampler(indices[:train_size])
        test_sampler = SubsetRandomSampler(indices[train_size:])

        train_loader = DataLoader(
            dataset, batch_size=64, sampler=train_sampler)
        test_loader = DataLoader(
            dataset, batch_size=64, sampler=test_sampler)

        # Each shadow_model is a PyTorch model
        # Train the shadow model
        if reload:
            print(f"[INFO] Loading shadow model {i}")
            # reload the saved model
            load_model(shadow_model, load_path=model_filename)
            print(f"[INFO] Done shadow model {i}")
        else:
            print(f"[INFO] Training shadow model {i}")
            train_model(shadow_model, train_loader,
                        device, num_epochs=num_epochs)
            save_model(model=shadow_model, save_path=model_filename)
            print(f"[INFO] Shadow model saved at {model_filename}")

        # Evaluate on training set
        print(f"[INFO] Evaluate on training set")
        shadow_model.eval()
        train_predictions = []
        train_labels = []
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = shadow_model(inputs)
                # _, predicted = torch.max(outputs, 1)
                # train_predictions.append(predicted)
                train_predictions.append(outputs)
                train_labels.append(labels)
        train_predictions = torch.cat(train_predictions)
        train_labels = torch.cat(train_labels)

        # Evaluate on testing set
        print(f"[INFO] Evaluate on test set")
        test_predictions = []
        test_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = shadow_model(inputs)
                # _, predicted = torch.max(outputs, 1)
                # test_predictions.append(predicted)
                test_predictions.append(outputs)
                test_labels.append(labels)
        test_predictions = torch.cat(test_predictions)
        test_labels = torch.cat(test_labels)

        member_samples.extend(indices[:train_size])
        member_true_label.append(train_labels)
        member_prediction.append(train_predictions)

        nonmember_samples.append(indices[train_size:])
        nonmember_true_label.append(test_labels)
        nonmember_prediction.append(test_predictions)

        train_acc = (torch.max(train_predictions, 1)[
                     1] == train_labels).sum() / len(train_labels)
        test_acc = (torch.max(test_predictions, 1)[
                    1] == test_labels).sum() / len(test_labels)
        print(f"[INFO] shadow model {i} train accuracy = {train_acc:.4f}")
        print(f"[INFO] shadow model {i} test accuracy = {test_acc:.4f}")

    # Concatenate the results of all the shadow models
    # all_member_samples = torch.cat(member_samples)
    all_member_true_label = torch.cat(member_true_label)
    all_member_prediction = torch.cat(member_prediction)
    # all_nonmember_samples = torch.cat(nonmember_samples)
    all_nonmember_true_label = torch.cat(nonmember_true_label)
    all_nonmember_prediction = torch.cat(nonmember_prediction)

    # encode to one-hot
    if all_member_true_label.shape != all_member_prediction.shape:
        all_member_true_label = nn.functional.one_hot(
            all_member_true_label).to(torch.float32)
        all_nonmember_true_label = nn.functional.one_hot(
            all_nonmember_true_label).to(torch.float32)

    return (
        (member_samples, all_member_true_label, all_member_prediction),
        (nonmember_samples, all_nonmember_true_label, all_nonmember_prediction),
    )


def MIA_shadow(member_prediction,
               nonmember_prediction,
               member_true_label,
               nonmember_true_label,
               device,
               batch_size=32,
               num_epoch=20,
               feature_dim=10,
               learning_rate=0.001,
               lr_scaling=False):
    """
    function for training attack model
    """
    features = member_prediction
    test_features = nonmember_prediction

    x_1 = torch.cat([features, test_features])
    x_2 = torch.cat([member_true_label, nonmember_true_label])

    # x_2 is the class in cifar-10, so 0-9
    # might need to do one-hot encoding
    # might want logit output instead of torch.max

    x_len = member_prediction.shape[0]
    test_len = nonmember_prediction.shape[0]
    # members
    labels = np.ones(x_len)
    # non-members
    test_labels = np.zeros(test_len)
    y_new = np.concatenate((labels, test_labels))

    attack_model = MembershipInferenceAttackModel(
        num_features=feature_dim, num_classes=feature_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(attack_model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(attack_model.parameters(), lr=learning_rate, momentum=0.9)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Assuming x_1, x_2, and y_new are numpy arrays or torch tensors
    attack_set = TensorDataset(x_1, x_2, torch.tensor(y_new))
    split_ratio = 0.9
    attack_train_set, attack_test_set = torch.utils.data.random_split(
        attack_set, [split_ratio, 1 - split_ratio], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        attack_train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        attack_test_set, batch_size=batch_size)

    attack_model = attack_model.to(device)  # Move model to GPU if available
    attack_model.train()

    for epoch in range(num_epoch):
        running_loss = 0.0
        running_acc = 0
        attack_model.train()
        for input1, input2, targets in train_loader:
            input1, input2, targets = input1.to(
                device), input2.to(device), targets.to(device)
            # input1: classifier output
            # input2: true class
            # targets: member vs non-member (0,1 classification)

            optimizer.zero_grad()
            outputs = attack_model(input1, input2)
            loss = criterion(outputs, targets.unsqueeze(1).float())
            
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += outputs.shape[0] * loss.item()
            running_acc += (torch.round(torch.flatten(outputs))
                            == targets).sum()

        # validation error
        attack_model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for input1, input2, targets in test_loader:
                input1, input2, targets = input1.to(
                    device), input2.to(device), targets.to(device)
                outputs = attack_model(input1, input2)
                val_acc += (torch.round(torch.flatten(outputs))
                            == targets).sum()
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):4f}, Train Acc:{running_acc/len(dataloader):4f}')
        print(f'Epoch {epoch+1}/{num_epoch}, Loss: {running_loss/len(train_loader.dataset): 4f},Train Acc: {running_acc*100/len(train_loader.dataset): 4f} Val Acc: {val_acc*100/len(test_loader.dataset): 4f}'
              )
        scheduler.step()

    # Evaluate final attack model
    attack_model.eval()
    running_acc = 0
    with torch.no_grad():
        for input1, input2, targets in train_loader:
            input1, input2, targets = input1.to(
                device), input2.to(device), targets.to(device)
            outputs = attack_model(input1, input2)
            # _, predicted = torch.max(outputs, 1)
            # train_predictions.append(predicted)
            running_acc += (torch.round(torch.flatten(outputs))
                            == targets).sum()
    running_acc = running_acc / len(train_loader.dataset)
    print(f"[INFO] attack model training accuracy = {running_acc}")
    return attack_model


# def attack_inference(target_model, attack_model, target_training_data, target_test_data, device,  batch_size = 32):
def attack_inference(target_model, attack_model, target_training_data, device,  batch_size=32, is_member=True):
    """
        :param attack_model: the attacker model for predicting whether (x,y) tuple is a member
        :param x: the input to target model
        :param y: the output of target model
    """
    target_model.eval()
    attack_model.eval()
    target_model.to(device)
    attack_model.to(device)

    member_loader = DataLoader(
        target_training_data, batch_size=batch_size, shuffle=False)
    # non_member_loader = DataLoader(target_test_data, batch_size=batch_size, shuffle=False)

    # making y_hat
    member_prediction, member_true_labels = model_predict(
        target_model, member_loader, device)
    # nonmember_prediction, nonmember_true_label = model_predict(target_model, non_member_loader, device)

    x_len = member_prediction.shape[0]
    # test_len = nonmember_prediction.shape[0]
    # members
    if is_member:
        labels = np.ones(x_len)
    else:
        labels = np.zeros(x_len)
    # non-members
    # test_labels = np.zeros(test_len)

    # x_1 = torch.cat([member_prediction, nonmember_prediction])
    # x_2 = torch.cat([member_true_labels, nonmember_true_label])
    # y_new = np.concatenate((labels, test_labels))

    # attack_data_set = TensorDataset(x_1, x_2, torch.tensor(y_new))
    attack_data_set = TensorDataset(
        member_prediction, member_true_labels, torch.tensor(labels))
    attack_loader = DataLoader(
        attack_data_set, batch_size=batch_size, shuffle=False)

    prediction = []
    member_labels = []
    with torch.no_grad():
        for input1, input2, targets in attack_loader:
            input1, input2, targets = input1.to(
                device), input2.to(device), targets.to(device)

            outputs = attack_model(input1, input2)
            prediction.append(outputs)
            member_labels.append(targets)

        prediction = torch.cat(prediction)
        member_labels = torch.cat(member_labels)

    return prediction, member_labels


def model_predict(model, data_loader, device, label_to_onehot=True):
    """
    Running model inference, used by attack_inference method
    :param model: Pytorch inference model (estimator/predictor)
    :param data_loader: Pytorch Dataloader for the dataset
    :param device: cpu or gpu to run the inference
    :param label_to_onehot: if true, encode the return true label to onehot 
    """
    model.to(device)
    prediction = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # _, predicted = torch.max(outputs, 1)
            # train_predictions.append(predicted)
            prediction.append(outputs)
            true_labels.append(labels)

    model_prediction = torch.cat(prediction)
    true_labels = torch.cat(true_labels)

    if label_to_onehot:
        true_labels = nn.functional.one_hot(true_labels).to(torch.float32)
    return model_prediction, true_labels


def MIA_evaluate(target_model,
                 attack_model,
                 target_traindata,
                 test_data, device,
                 result_save_location,
                 save_result=True,
                 append_result=True):

    return 
    # run attack inference on member data
    member_infer, member_labels = attack_inference(
        target_model, attack_model, target_traindata, device, is_member=True)
    member_accuracy = torch.round(member_infer).sum() / len(member_labels)

    # run attack inference on non-member data
    nonmember_infer, non_member_labels = attack_inference(
        target_model, attack_model, test_data, device, is_member=False)
    non_member_accuracy = torch.round(
        1 - nonmember_infer).sum() / len(non_member_labels)

    print(f"[RESULT] member attack accuracy {member_accuracy:.4f}")
    print(f"[RESULT] non member attack accuracy {non_member_accuracy:.4f}")

    # save the attack accuracy result
    if save_result:
        # result_save_location = "./output/"
        result_filename = "attack_result.txt"
        make_if_not_exist(result_save_location)
        if append_result:
            with open(os.path.join(result_save_location, result_filename), "a") as f:
                f.write(f"==============================")
                f.write(f"Attack on original model")
                f.write(f"member attack accuracy {member_accuracy:.4f}")
                f.write(f"nonmember attack accuracy {non_member_accuracy:.4f}")
        else:
            with open(os.path.join(result_save_location, result_filename), "w") as f:
                f.write(f"==============================")
                f.write(f"Attack on original model")
                f.write(f"member attack accuracy {member_accuracy:.4f}")
                f.write(f"nonmember attack accuracy {non_member_accuracy:.4f}")

    return member_accuracy, non_member_accuracy

def unlearn_eval(model, member_data, non_member_data=None, device=None, batch_size=64):
    """
    Running loss, accuracy, and model confidence on member vs non-member
    """
    criterion = nn.CrossEntropyLoss()
    member_loader = DataLoader(member_data, batch_size=batch_size, shuffle=False)
    nonmember_loader = DataLoader(non_member_data, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    ### run loss
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    running_conf = 0.0

    softmax = nn.Softmax(1)
    with torch.no_grad():
        for input, target in member_loader:
            input, target = input.to(device), target.to(device)
            ### calculate model output
            outputs = model(input)

            ### calculate model loss
            loss = criterion(outputs, target)

            ### calculate model prediction
            prob_vec = softmax(outputs)
            max_conf, max_class = torch.max(prob_vec, 1)

            running_loss += outputs.shape[0] * loss.item()
            running_acc += (max_class == target).sum().item()
            running_conf += max_conf.sum().item()

    ### save the evaluation data on member data
    member_loss = running_loss / len(member_data)
    member_acc = running_acc / len(member_data)
    member_avg_conf = running_conf / len(member_data)


    ### Evaluate on non-member data
    with torch.no_grad():
        ### reset the stat
        running_loss = 0.0
        running_acc = 0.0
        running_conf = 0.0
        for input, target in nonmember_loader:
            input, target = input.to(device), target.to(device)
            ### calculate model output
            outputs = model(input)

            ### calculate model loss
            loss = criterion(outputs, target)

            ### calculate model prediction
            prob_vec = softmax(outputs)
            max_conf, max_class = torch.max(prob_vec, 1)

            running_loss += outputs.shape[0] * loss.item()
            running_acc += (max_class == target).sum().item()
            running_conf += max_conf.sum().item()

    nonmember_loss = running_loss / len(non_member_data)
    nonmember_acc = running_acc / len(non_member_data)
    nonmember_avg_conf = running_conf / len(non_member_data)

    return (member_loss, member_acc, member_avg_conf), (nonmember_loss, nonmember_acc, nonmember_avg_conf)

def unlearn_eval_list(model_list, target_train_list, test_data, device, batch_size=64):
    total_member_loss = 0
    total_member_acc = 0
    total_member_avg_conf = 0
    total_nonmember_loss = 0
    total_nonmember_acc = 0
    total_nonmember_avg_conf = 0

    totol_member_count = 0
    totol_nonmember_count = 0
    for i, model in enumerate(model_list):
        member_stat, non_member_stat = unlearn_eval(model, member_data=target_train_list[i], non_member_data=test_data, device=device, batch_size=batch_size)
        member_loss, member_acc, member_avg_conf = member_stat
        nonmember_loss, nonmember_acc, nonmember_avg_conf = non_member_stat

        ### adjust the stat by length of data
        member_length = len(target_train_list[i])
        non_member_length = len(test_data)
        totol_member_count += member_length
        totol_nonmember_count += non_member_length

        total_member_loss += member_loss * member_length
        total_member_acc += member_acc * member_length
        total_member_avg_conf += member_avg_conf * member_length
        total_nonmember_loss += nonmember_loss * non_member_length
        total_nonmember_acc += nonmember_acc * non_member_length
        total_nonmember_avg_conf += nonmember_avg_conf * non_member_length

    total_member_loss = total_member_loss / totol_member_count
    total_member_acc = total_member_acc / totol_member_count
    total_member_avg_conf = total_member_avg_conf / totol_member_count
    total_nonmember_loss = total_nonmember_loss / totol_nonmember_count
    total_nonmember_acc = total_nonmember_acc / totol_nonmember_count
    total_nonmember_avg_conf = total_nonmember_avg_conf / totol_nonmember_count

    return (total_member_loss, total_member_acc, total_member_avg_conf), (total_nonmember_loss, total_nonmember_acc, total_nonmember_avg_conf)

def naive_unlearn(retain_dataset, device, num_classes = 10):
    model = get_model(num_classes)
    dataloader = DataLoader(retain_dataset, batch_size=64, shuffle=True)
    train_model(model, dataloader, device, num_epochs=10)

    return model

#########################################################
### following code are unused
def membership_inference_attack(model, t_loader, device, f_loader=None, seed=42):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if f_loader is not None:
        fgt_cls = list(np.unique(f_loader.dataset.targets))
        indices = [i in fgt_cls for i in t_loader.dataset.targets]
        t_loader.dataset.data = t_loader.dataset.data[indices]
        t_loader.dataset.targets = t_loader.dataset.targets[indices]

    criterion = nn.CrossEntropyLoss(reduction='none')
    test_losses = []
    forget_losses = []
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        t_loader.dataset, batch_size=128, shuffle=False)

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)
        test_losses = test_losses + list(loss.cpu().detach().numpy())
    del dataloader

    if f_loader is not None:
        dataloader = torch.utils.data.DataLoader(
            f_loader.dataset, batch_size=128, shuffle=False)
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            forget_losses = forget_losses + list(loss.cpu().detach().numpy())
        del dataloader

    np.random.seed(seed)
    random.seed(seed)
    if f_loader is not None:
        if len(forget_losses) > len(test_losses):
            forget_losses = list(random.sample(
                forget_losses, len(test_losses)))
        elif len(test_losses) > len(forget_losses):
            test_losses = list(random.sample(test_losses, len(forget_losses)))

    sns.distplot(np.array(test_losses), kde=False, norm_hist=False,
                 rug=False, label='test-loss', ax=plt)
    if f_loader is not None:
        sns.distplot(np.array(forget_losses), kde=False,
                     norm_hist=False, rug=False, label='forget-loss', ax=plt)
    plt.legend(prop={'size': 14})
    plt.tick_params(labelsize=12)
    plt.title("loss histograms", size=18)
    plt.xlabel('loss values', size=14)
    plt.show()
    print(np.max(test_losses), np.min(test_losses))
    if f_loader is not None:
        print(np.max(forget_losses), np.min(forget_losses))

    test_labels = [0]*len(test_losses)
    forget_labels = [1]*len(forget_losses)
    features = np.array(test_losses + forget_losses).reshape(-1, 1)
    labels = np.array(test_labels + forget_labels).reshape(-1)
    features = np.clip(features, -100, 100)
    score = evaluate_attack_model(
        features, labels, n_splits=5, random_state=seed)

    return score


def evaluate_attack_model(sample_loss,
                          members,
                          n_splits=5,
                          random_state=None):
    """Computes the cross-validation score of a membership inference attack.
    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
      random_state: int, RandomState instance or None, default=None
        random state to use in cross-validation splitting.
    Returns:
      score : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = LogisticRegression()
    cv = StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state)
    return cross_val_score(attack_model, sample_loss, members, cv=cv, scoring=cm_score)



def all_readouts(model, thresh=0.1, name='method', seed=0):
    MIA = membership_inference_attack(
        model, copy.deepcopy(test_loader_full), forget_loader, seed)
    # train_loader = torch.utils.data.DataLoader(train_loader_full.dataset, batch_size=128, shuffle=True)
    # readout_retrain(model, train_loader, forget_loader, epochs=100, lr=0.001, threshold=thresh)
    retrain_time, _ = 0, 0
    test_error = test(model, test_loader_full)['error']*100
    forget_error = test(model, forget_loader)['error']*100
    retain_error = test(model, retain_loader)['error']*100
    val_error = test(model, valid_loader_full)['error']*100

    print(f"{name} ->"
          f"\tFull test error: {test_error:.2f}"
          f"\tForget error: {forget_error:.2f}\tRetain error: {retain_error:.2f}\tValid error: {val_error:.2f}"
          f"\tFine-tune time: {retrain_time+1} steps\tMIA: {np.mean(MIA):.2f}±{np.std(MIA):0.1f}")

    return (dict(test_error=test_error, forget_error=forget_error, retain_error=retain_error, val_error=val_error, retrain_time=retrain_time+1, MIA=np.mean(MIA)))

