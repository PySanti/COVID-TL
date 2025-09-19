import torchvision
from torchvision.models import  resnet18
import torch

def load_model(num_class):
    model = torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, num_class)  # 2 clases: COVID / No-COVID
    return model
