import torch
import os
import numpy as np
import torch.utils
import torch.utils.data

from model import get_model
# adversarial eval
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier


def load_models(num_shards, num_classes, save_directory):
    models = []
    for i in range(num_shards):
        model_filename = os.path.join(save_directory, f"model_shard_{i}.pt")
        model = get_model(num_classes)
        model.load_state_dict(torch.load(model_filename))
        models.append(model)
    return models


def evaluate_majority_vote(models, test_loader, device):
    correct = 0
    total = 0
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = []

            # Get predictions from each model
            for model in models:
                model.eval()
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted.cpu().numpy())

            # Aggregate predictions using majority vote
            majority_vote = np.array(predictions).mean(axis=0)
            majority_vote = np.round(majority_vote).astype(int)
            all_predictions.extend(majority_vote)

            # Calculate accuracy
            total += labels.size(0)
            correct += (majority_vote == labels.cpu().numpy()).sum().item()

    accuracy = 100 * correct / total
    return accuracy, all_predictions

# ART model evaluation, replacing it's own predict method


def new_predict(
    self, x_loader: torch.utils.data.DataLoader, batch_size: int = 128, training_mode: bool = False
):
    # Set model mode

    self._model.train(mode=training_mode)
    results_list = []
    with torch.no_grad():
        for (x_batch, x_label) in x_loader:
            # Move inputs to device
            x_batch = x_batch.to(self._device)

            # Run prediction
            model_outputs = self._model(x_batch)
            output = model_outputs[-1]
            output = output.detach().cpu().numpy().astype(np.float32)
            if len(output.shape) == 1:
                output = np.expand_dims(output, axis=1).astype(np.float32)

            results_list.append(output)

        results = np.vstack(results_list)

    return results


# adversarial eval
def eval_adversarial(model, x_test):
    criterion = nn.CrossEntropyLoss()

    # might need to do model conversion first
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    # Step 5: Evaluate the ART classifier on benign test examples

    predictions = model.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) ==
                      np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Step 6: Generate adversarial test examples
    attack = FastGradientMethod(estimator=model, eps=0.2)
    x_test_adv = attack.generate(x=x_test)

    # Step 7: Evaluate the ART model on adversarial test examples

    predictions = model.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) ==
                      np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
