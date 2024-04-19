### basic functions for pytorch training loop

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

def collate_fn(batch):
    """
    Custom collate function to handle Subset objects
    """
    if isinstance(batch[0], Subset):
        return [sample for subset in batch for sample in subset]
    else:
        return torch.utils.data.dataloader.default_collate(batch)
    
def train_model(model, dataloader, device, num_epochs=10):
    model.to(device)
    # Define your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0
        for inputs, labels in dataloader:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += outputs.shape[0] * loss.item()
            if labels.shape != outputs.shape:
                _, predicted = torch.max(outputs, 1)
                running_acc += (labels==predicted).sum().item()
            else:
                running_acc += (labels==outputs).sum().item()

        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):4f}, Train Acc:{running_acc/len(dataloader):4f}')
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):4f}, Train Acc:{running_acc*100/len(dataloader.dataset):4f}')

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path))
    return model

