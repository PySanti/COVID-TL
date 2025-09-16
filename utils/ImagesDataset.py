from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms



class ImagesDataset(Dataset):
    """
        Wrapper para las rutas y labels de imagenes
        para ser usado con dataloader de entrenamiento,
        validacion y pruebas
    """
    def __init__(self, X, Y, transformer) -> None:
        super().__init__()
        self.X = X
        self.Y = Y
        self.transformer = transformer


    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        image = Image.open(self.X[idx]).convert('RGB')
        trans_image = self.transformer(image)
        image.close()
        return trans_image, self.Y[idx]
