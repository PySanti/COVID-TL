import torchvision
from torchvision.models import  resnet18
import torch

def load_model(num_class):
    model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_class)  # 2 clases: COVID / No-COVID
    return model
