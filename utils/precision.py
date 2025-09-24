import torch
from sklearn.metrics import precision_score

def precision(output, Y_batch):
    _, prediction = torch.max(output, 1)
    return (Y_batch == prediction).to("cpu").sum() / len(Y_batch)
