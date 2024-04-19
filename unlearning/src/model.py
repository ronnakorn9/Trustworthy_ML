import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def get_model(num_classes):
    """
    Create Pytorch's resnet50 model with specified number of predictor class
    """
    print("[INFO] loading model")
    # Initialize the pre-trained ResNet model
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    print("[INFO] done loading model")
    # Freeze the parameters of the pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify the last fully connected layer to fit the number of classes in your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model