"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import copy
import os
import numpy as np
import random
import time

from art.attacks.evasion import FastGradientMethod
from art.attacks.inference.membership_inference import ShadowModels, MembershipInferenceBlackBox
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist, load_cifar10


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from eval import load_models, evaluate_majority_vote, new_predict, load_datasets
from train import train_model, save_model
from art_edit import generate_shadow_dataset, MIA_shadow, attack_inference, MembershipInferenceAttackModel, MIA_evaluate, unlearn_eval, unlearn_eval_list, naive_unlearn
from SCRUB import SCRUB_unlearn, scrub
from SCRUB_util import get_dataloaders
from util import make_if_not_exist


# Step 1: Load the CIFAR dataset
seed = 42
new_data = False
num_shards = 3
num_classes = 10
data_dir = f"./data/shard_{num_shards}/split_data/"
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    # transform=ToTensor()
    transform=preprocess
)

if new_data:
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        # transform=ToTensor()
        transform=preprocess
    )
    
    # Step 1.1: setup split index
    split_ratio = 0.5
    target_train, shadow_train = torch.utils.data.random_split(training_data, [split_ratio, 1 - split_ratio], generator=torch.Generator().manual_seed(42))

    torch.save(target_train, './data/cifar-10_split/target_train.pt')
    torch.save(shadow_train, './data/cifar-10_split/shadow_train.pt')
else:
    train_data_shards = load_datasets(num_shards, data_dir)

############################################
# Step 2: Create the model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reload = True
# save_path = "./output/example/model/"
save_path = f"./output/shard_{num_shards}/model/"
model_list = []
model_list = load_models(num_shards, num_classes, save_directory=save_path)

### this is for non-sharded model
# model_name = "model_shard_.pt"
# model_name = "resnet50.pt"
# model_location = os.path.join(save_path, model_name)
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# for param in model.parameters():
#     param.requires_grad = False
# model.fc = nn.Linear(model.fc.in_features, 10)

# if reload:
#     model.load_state_dict(torch.load(model_location))
# else:
#     exit()
#     model.to(device)
#     dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
#     train_model(model, dataloader, device, num_epochs=20)
#     torch.save(model.state_dict(), model_location)

print("[INFO] done loading model")



############################################
# Step 3: Train shadow models + generate attack training data
use_existing_attack = True
num_shadow = 5
attack_model_location = "./model/attack/"
attack_model_name = f"nn_attack_{num_shadow}_shadow.pt"
attack_full_path = os.path.join(attack_model_location, attack_model_name)

if not use_existing_attack:
    # Traning shadow model
    shadow_models = []
    shadow_epoch = 20
    shadow_reload = True
    data_reload = True
    if not data_reload:
        for i in range(num_shadow):
            shadow_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            for param in shadow_model.parameters():
                param.requires_grad = False
            shadow_model.fc = nn.Linear(shadow_model.fc.in_features, num_classes)
            shadow_models.append(shadow_model)

        # Traning shadow model and generate data for attack model
        shadow_result = generate_shadow_dataset(dataset=shadow_train, shadow_models=shadow_models, num_epochs=shadow_epoch, device=device, reload=shadow_reload)
        (member_x, member_y, member_predictions), (nonmember_x, nonmember_y, nonmember_predictions) = shadow_result

        make_if_not_exist("./data/attack_data/")
        torch.save(member_predictions, './data/attack_data/member_predictions.pt')
        torch.save(nonmember_predictions, './data/attack_data/nonmember_predictions.pt')
        torch.save(member_y, './data/attack_data/member_y.pt')
        torch.save(nonmember_y, './data/attack_data/nonmember_y.pt')
    else: 
        member_predictions = torch.load('./data/attack_data/member_predictions.pt')
        nonmember_predictions = torch.load('./data/attack_data/nonmember_predictions.pt')
        member_y = torch.load('./data/attack_data/member_y.pt')
        nonmember_y = torch.load('./data/attack_data/nonmember_y.pt')

    ############################################
    # Step 4: Train attack model

    print("[INFO] training attack model")
    attack_model = MIA_shadow(member_predictions,
                                nonmember_predictions,
                                member_y,
                                nonmember_y,
                                device,
                                batch_size= 128,
                                num_epoch= 20,
                                feature_dim= 10,
                                learning_rate=1e-4)
    
    print("[INFO] done training attack model")
    
    ### save atack model
    make_if_not_exist(attack_model_location)
    save_model(attack_model, attack_full_path)
    print(f"[INFO] attack model saved at {attack_full_path}")

else:
    ### load atack model
    attack_model = MembershipInferenceAttackModel(num_features=num_classes, num_classes=num_classes)
    attack_model.load_state_dict(torch.load(attack_full_path))
    print("[INFO] done loading attack model")

############################################
# Step 5: Evaluate original model 
print("[INFO] Evaluate original model...")
# member_stat, non_member_stat =  unlearn_eval(model, target_train, test_data, device, batch_size=64)

member_stat, non_member_stat =  unlearn_eval_list(model_list, train_data_shards, test_data, device, batch_size=64)
member_loss, member_acc, member_avg_conf = member_stat
nonmember_loss, nonmember_acc, nonmember_avg_conf = non_member_stat
print("======== Original model stat ========")
print(f"[RESULT] member average loss = {member_loss}")
print(f"[RESULT] member average accuracy = {member_acc}")
print(f"[RESULT] member average confidence = {member_avg_conf}")
print("=====================================")
print(f"[RESULT] non-member average loss = {nonmember_loss}")
print(f"[RESULT] non-member average accuracy = {nonmember_acc}")
print(f"[RESULT] non-member average confidence = {nonmember_avg_conf}")
print("=====================================")

# Step 5.1: prepare forget & retain data
### scenario 1: 100 forget point in one shard
num_to_forget = 100
target_train = train_data_shards[0]
original_model = copy.deepcopy(model_list[0])
(forget_loader, retain_loader), (forget_dataset, retain_dataset) = get_dataloaders(target_train, num_to_forget)


### Evaluate MIA on original model (doesn't work, attack model too weak)
MIA_evaluate(target_model = original_model, 
             attack_model = attack_model, 
             target_traindata = target_train, 
             test_data = test_data, 
             device = device, 
             result_save_location = save_path,
             save_result = True, append_result=False)

MIA_evaluate(target_model = original_model, 
             attack_model = attack_model, 
             target_traindata = retain_dataset, 
             test_data = forget_dataset, 
             device = device, 
             result_save_location = save_path,
             save_result = True, append_result=True)
print("[INFO] done Evaluate attack model on original model")
############################################
# Step 6: Unlearn (naive) and evaluate model 

print(f"\n[INFO] unlearn scenario: {num_to_forget} data points\n")

print(f"[INFO] naive unlearn on {num_to_forget} data points")
start = time.time()
naive_unlearn_model = naive_unlearn(retain_dataset, device, num_classes = 10)
end = time.time()
print(f"[RESULT] naive unlearn time: {end - start:.4f} seconds")
train_data_shards[0] = retain_dataset
model_list[0] = naive_unlearn_model

# member_stat, non_member_stat =  unlearn_eval(naive_unlearn_model, retain_dataset, forget_dataset, device, batch_size=64)
member_stat, non_member_stat =  unlearn_eval_list(model_list, train_data_shards, test_data, device, batch_size=64)

member_loss, member_acc, member_avg_conf = member_stat
nonmember_loss, nonmember_acc, nonmember_avg_conf = non_member_stat
print(f"======== {num_to_forget} point naive forget model stat ========")
print(f"[RESULT] retain data average loss = {member_loss}")
print(f"[RESULT] retain data average accuracy = {member_acc}")
print(f"[RESULT] retain data average confidence = {member_avg_conf}")
print("=====================================")
print(f"[RESULT] forget data average loss = {nonmember_loss}")
print(f"[RESULT] forget data average accuracy = {nonmember_acc}")
print(f"[RESULT] forget data average confidence = {nonmember_avg_conf}")

############################################
# Step 7: Unlearn (SCRUB) and evaluate model 

print("[INFO] SCRUB unlearning ...")
reuse_scrub = False
unlearned_model_rewind, unlearned_model = scrub(original_model, forget_loader, retain_loader, reuse_scrub=reuse_scrub, name_append=num_to_forget)
model_list[0] = unlearned_model
# member_stat, non_member_stat =  unlearn_eval(unlearned_model, retain_dataset, forget_dataset, device, batch_size=64)
print("[INFO] Evaluate SCRUB model...")
member_stat, non_member_stat =  unlearn_eval_list(model_list, train_data_shards, test_data, device, batch_size=64)
member_loss, member_acc, member_avg_conf = member_stat
nonmember_loss, nonmember_acc, nonmember_avg_conf = non_member_stat
print(f"======== {num_to_forget} point SCRUB forget model stat ========")
print(f"[RESULT] retain data average loss = {member_loss}")
print(f"[RESULT] retain data average accuracy = {member_acc}")
print(f"[RESULT] retain data average confidence = {member_avg_conf}")
print("=====================================")
print(f"[RESULT] forget data average loss = {nonmember_loss}")
print(f"[RESULT] forget data average accuracy = {nonmember_acc}")
print(f"[RESULT] forget data average confidence = {nonmember_avg_conf}")

# Step 6.1: Evaluate MIA on SCRUB unlearn model [DOESN'T WORK]
MIA_evaluate(target_model = unlearned_model, 
             attack_model = attack_model, 
             target_traindata = target_train, 
             test_data = test_data, 
             device = device, 
             result_save_location = save_path,
             save_result = True, append_result=True)

MIA_evaluate(target_model = unlearned_model, 
             attack_model = attack_model, 
             target_traindata = retain_dataset, 
             test_data = forget_dataset, 
             device = device, 
             result_save_location = save_path,
             save_result = True, append_result=True)
print("[INFO] done Evaluate attack model on SCRUB model")
############################################
### Forget set 2

print(f"\n[INFO] unlearn scenario: {num_to_forget} data points\n")
num_to_forget = 1000


(forget_loader, retain_loader), (forget_dataset, retain_dataset) = get_dataloaders(target_train, num_to_forget)
print(f"[INFO] naive unlearn on {num_to_forget} data points")
start = time.time()
naive_unlearn_model = naive_unlearn(retain_dataset, device, num_classes = 10)
end = time.time()
print(f"[RESULT] naive unlearn time: {end - start:.4f} seconds")
train_data_shards[0] = retain_dataset
model_list[0] = naive_unlearn_model

# member_stat, non_member_stat =  unlearn_eval(naive_unlearn_model, retain_dataset, forget_dataset, device, batch_size=64)
member_stat, non_member_stat =  unlearn_eval_list(model_list, train_data_shards, test_data, device, batch_size=64)

member_loss, member_acc, member_avg_conf = member_stat
nonmember_loss, nonmember_acc, nonmember_avg_conf = non_member_stat
print(f"======== {num_to_forget} point naive forget model stat ========")
print(f"[RESULT] retain data average loss = {member_loss}")
print(f"[RESULT] retain data average accuracy = {member_acc}")
print(f"[RESULT] retain data average confidence = {member_avg_conf}")
print("=====================================")
print(f"[RESULT] forget data average loss = {nonmember_loss}")
print(f"[RESULT] forget data average accuracy = {nonmember_acc}")
print(f"[RESULT] forget data average confidence = {nonmember_avg_conf}")

reuse_scrub = False
print("[INFO] SCRUB unlearning ...")

unlearned_model_rewind, unlearned_model = scrub(original_model, forget_loader, retain_loader, reuse_scrub=reuse_scrub, name_append=num_to_forget)
model_list[0] = unlearned_model

print("[INFO] Evaluate SCRUB model...")
# member_stat, non_member_stat =  unlearn_eval(unlearned_model, retain_dataset, forget_dataset, device, batch_size=64)
member_stat, non_member_stat =  unlearn_eval_list(model_list, train_data_shards, test_data, device, batch_size=64)

member_loss, member_acc, member_avg_conf = member_stat
nonmember_loss, nonmember_acc, nonmember_avg_conf = non_member_stat
print(f"======== {num_to_forget} point SCRUB forget model stat ========")
print(f"[RESULT] retain data average loss = {member_loss}")
print(f"[RESULT] retain data average accuracy = {member_acc}")
print(f"[RESULT] retain data average confidence = {member_avg_conf}")
print("=====================================")
print(f"[RESULT] forget data average loss = {nonmember_loss}")
print(f"[RESULT] forget data average accuracy = {nonmember_acc}")
print(f"[RESULT] forget data average confidence = {nonmember_avg_conf}")

# MIA_evaluate(target_model = unlearned_model, 
#              attack_model = attack_model, 
#              target_traindata = target_train, 
#              test_data = test_data, 
#              device = device, 
#              result_save_location = save_path,
#              save_result = True, append_result=True)

# MIA_evaluate(target_model = unlearned_model, 
#              attack_model = attack_model, 
#              target_traindata = retain_dataset, 
#              test_data = forget_dataset, 
#              device = device, 
#              result_save_location = save_path,
#              save_result = True, append_result=True)

print("[INFO] all done")