### for spliting datasett into shards

from torch.utils.data import Dataset, Subset, DataLoader
from math import ceil
import json
import os
import configparser
import pickle

def read_config(config_file):
    print(f"[INFO] reading config from \'{config_file}\'")
    config = configparser.ConfigParser()
    config.read(config_file)
    dataset_location = config['Dataset']['dataset_location']
    save_directory = config['Dataset']['save_directory']
    num_shards = int(config['Dataset']['num_shards'])
    num_slices = int(config['Dataset']['num_slices'])

    # Create save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    return dataset_location, save_directory, num_shards, num_slices

def save_shard_mapping(save_directory, shard_mapping):
    json_file = os.path.join(save_directory, 'shard_mapping.json')
    with open(json_file, 'w') as f:
        json.dump(shard_mapping, f, indent=4)

def load_shards(directory):
    print(f"[INFO] loading shards from {directory}")
    if not os.path.exists(directory):
        # no data to load
        return None
    shards = []
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            shard_filename = os.path.join(directory, filename)
            with open(shard_filename, 'rb') as f:
                shard = pickle.load(f)
            shards.append(shard)
    return shards

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def split_dataset(dataset, num_shards, num_slices, printing=True):
    if printing:
        print(f"[INFO] spiting dataset into {num_shards} shards")
    shard_size = ceil(len(dataset) / num_shards)
    shards = []
    shard_index_to_dataset_index = {}
    dataset_index = 0
    for i in range(num_shards):
        if printing:
            print(f"[INFO] making shard {i}")
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, len(dataset))
        shard_indices = list(range(start_idx, end_idx))
        shard = Subset(dataset, shard_indices)

        # Divide the shard into slices
        slice_size = ceil(len(shard) / num_slices)
        # slices = []
        shard_index_to_dataset_index[i] = []
        for j in range(num_slices):
            start_slice_idx = j * slice_size
            end_slice_idx = min((j + 1) * slice_size, len(shard))
            slice_indices = list(range(start_slice_idx, end_slice_idx))
            # slice_subset = Subset(shard, slice_indices)
            # slices.append(slice_subset)
            shard_index_to_dataset_index[i].append({i : slice_indices})
        shards.append(shard)
        # shard_index_to_dataset_index[i] = shard_indices

    return shards, shard_index_to_dataset_index

# Example usage:
if __name__ == "__main__":
    config_file = 'config/config.ini'  # Path to your config file
    dataset_location, save_directory, num_shards, num_slices = read_config(config_file)
    print("[INFO] Dataset Location:", dataset_location)
    print("[INFO] Save Directory:", save_directory)
    print("[INFO] Number of Shards:", num_shards)


    # Assuming you have your dataset stored in a list called 'data'
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # Replace this with your dataset
    custom_dataset = CustomDataset(data)

    shards, shard_mapping = split_dataset(custom_dataset, num_shards, num_slices)

    # Example of accessing the shards
    for i, shard in enumerate(shards):
        print(f"Shard {i+1}: {list(shard)}")

    # Example of accessing the shard index to dataset index mapping
    print("Shard index to dataset index mapping:")
    print(shard_mapping)

    # Save shard_index_to_dataset_index to JSON
    save_shard_mapping(save_directory, shard_mapping)