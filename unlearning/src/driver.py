import os
import pickle

from train import train_model, collate_fn, save_model, load_model
from shard import read_config, save_shard_mapping, split_dataset, load_shards, CustomDataset
from model import get_model
from eval import load_models, evaluate_majority_vote
# for testing
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

import random
import numpy as np
import time
def make_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def train_procedure(shards, device, split_model_dir):
    model_list = []
    for i, shard in enumerate(shards):
        print(f'Training model for shard {i+1}...')
        print(type(shard))

        # Get the model for this shard
        num_classes = 10  # Change this according to your dataset
        model = get_model(num_classes)
        model.to(device)
        
        # Create a DataLoader for the shard
        dataloader = DataLoader(shard, batch_size=64, shuffle=True, collate_fn=collate_fn)
        
        # Train the model with the shard
        train_model(model, dataloader, device)
        
        # Save the trained model
        model_filename = os.path.join(split_model_dir, f"model_shard_{i}.pt")
        
        model_list.append(model)
        save_model(model, model_filename)
        print(f"Trained model for shard {i} saved to {model_filename}")
    return model_list


def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ### 1. load the config
    config_file = 'config/config.ini'  # Path to your config file
    dataset_location, save_directory, num_shards, num_slices = read_config(config_file)
    print("[INFO] Dataset Location:", dataset_location)
    print("[INFO] Save Directory:", save_directory)
    print("[INFO] Number of Shards:", num_shards)
    split_model_dir = os.path.join(save_directory, f"model")

    make_if_not_exist(save_directory)
    make_if_not_exist(split_model_dir)

    ### 2. load the dataset
    save_data = True
    split_data_dir = os.path.join(dataset_location, f"split_data")
    shards = load_shards(split_data_dir)
    
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if shards is None:
        ### no previously split, load a new one
        print("[INFO] No previous sharded data, making new sharded data")
        # Assuming you have your dataset stored in a list called 'data'
        

        training_data_full = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            # transform=ToTensor()
            transform=preprocess
        )
        split_ratio = 0.5
        training_data, _ = torch.utils.data.random_split(training_data_full, [split_ratio, 1 - split_ratio])

        print("[INFO] done loading dataset")
        ### 3. shard/split the data
        shards, shard_mapping = split_dataset(training_data, num_shards, num_slices)
        print("[INFO] done sharding dataset")
        if save_data:
            # Create save directory if it doesn't exist
            make_if_not_exist(split_data_dir)

            for i, shard in enumerate(shards):
                shard_filename = os.path.join(split_data_dir, f"shard_{i}.pkl")
                with open(shard_filename, 'wb') as f:
                    pickle.dump(shard, f)
                print(f"Shard {i} saved to {shard_filename}")

        # Save shard_index_to_dataset_index to JSON
        save_shard_mapping(save_directory, shard_mapping)


    ### 4. load the model
    num_classes = 10  # Change this according to dataset, CIFAR-10 has 10 classes
    training_time = None
    ### 5. train/load model
    model_list = []
    try:
        model_list = load_models(num_shards, num_classes, split_model_dir)
    except:
        print("[ERROR] no previously trained model. Training new model")
        start = time.time()
        model_list = train_procedure(shards, device, split_model_dir)
        end = time.time()
        training_time = end - start
        
    
    ### 6. test
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=preprocess
    )
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Evaluate using majority vote
    accuracy, all_predictions = evaluate_majority_vote(model_list, test_loader, device)
    print(f"Test accuracy using majority vote: {accuracy:.2f}%")

    result_filename = "result.txt"
    with open(os.path.join(save_directory, result_filename), "w") as f:
        f.write(f"{num_shards} Shard model")
        f.write(f"Accuracy using majority vote: {accuracy:.2f}%")
        if training_time is not None:
            f.write(f"training time: {training_time:.4f} seconds")


if __name__ == "__main__":
    main()