from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms



class ImagesDataset(Dataset):
    """
        Wrapper para las rutas y labels de imagenes
        para ser usado con dataloader de entrenamiento,
        validacion y pruebas
    """
    def __init__(self, X, Y, train_transformer, val_transformer, train_dataset : bool, minority_class) -> None:
        super().__init__()
        self.X = X
        self.Y = Y
        self.val_transformer = val_transformer
        self.train_transformer = train_transformer
        self.minority_class = minority_class
        self.train_dataset = train_dataset


    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        image = Image.open(self.X[idx]).convert('RGB')
        trans_image = self.train_transformer(image) if self.train_dataset else self.val_transformer(image)
        image.close()
        return trans_image, self.Y[idx]
