from torchvision import transforms
import numpy as np
from PIL import Image
import torch

class NormDataset(torch.utils.data.Dataset):
    """
        Dataset creado para wrappear los
        paths_sets para encontrar los valores de normalizacion
        en el trainset (media y desviacion estandar por canal)
    """

    def __init__(self, X) -> None:
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = Image.open(self.X[idx]).convert('RGB')
        image_arr = np.array(image)
        image.close()
        return image_arr

