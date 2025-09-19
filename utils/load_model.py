import torchvision
from torchvision.models import  resnet18
import torch

def load_model(num_class):
    model = torchvision.models.squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT)

#    for param in model.parameters():
#        param.requires_grad = False

    model.classifier[1] = torch.nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
    model.num_classes = 2
    return model
