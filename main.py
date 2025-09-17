import kagglehub
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.NormDataset import NormDataset
from utils.load_dataset import load_dataset
from utils.load_model import load_model
from utils.normalization_metrics_calc import normalization_metrics_calc
from utils.ImagesDataset import ImagesDataset
from torchvision import transforms
from utils.MACROS import BATCH_SIZE, EPOCHS, IMAGE_SIZE, MEANS, STDS
import torchvision

if __name__ == "__main__":

    data_path = kagglehub.dataset_download("andyczhao/covidx-cxr2")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_transformer  = transforms.Compose([

        # basic resize
        
        transforms.Resize((IMAGE_SIZE[0], IMAGE_SIZE[1])), # redimensionar 
        transforms.CenterCrop((IMAGE_SIZE[0]*0.9, IMAGE_SIZE[1]*0.9)), # recordar desde el centro para consistencia
        
        # data augmentation

        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomRotation(15),


        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS)
    ])

    val_transformer = transforms.Compose([
        transforms.Resize((IMAGE_SIZE[0], IMAGE_SIZE[1])), # redimensionar 
        transforms.CenterCrop((IMAGE_SIZE[0]*0.9, IMAGE_SIZE[1]*0.9)), # recordar desde el centro para consistencia
        

        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS)
    ])

    trainset = ImagesDataset(*load_dataset('train', data_path), train_transformer)
    valset = ImagesDataset(*load_dataset('val', data_path), val_transformer)
    testset = ImagesDataset(*load_dataset('test', data_path), val_transformer)


    trainloader = DataLoader(trainset, BATCH_SIZE, shuffle=True, num_workers=12, persistent_workers=True, pin_memory=True)
    valloader = DataLoader(valset, BATCH_SIZE, shuffle=False, num_workers=5, persistent_workers=True, pin_memory=True)

    model = load_model(2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",patience=8)

    for i in range(EPOCHS):

        batches_train_loss = []
        batches_val_loss = []
        batches_val_prec =[]
        
        model.train()
        for a, (X_batch, Y_batch) in enumerate(trainloader):
            print(f"train batch : {a}/{len(trainloader)}", end="\r")
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

            optimizer.zero_grad()

            output = model(X_batch)
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()


            batches_train_loss.append(loss.item())

        print("\n\n")

        model.eval()

        with torch.no_grad():
            for a, (X_batch, Y_batch) in enumerate(valloader):
                print(f"val batch : {a}/{len(valloader)}", end="\r")
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

                output = model(X_batch)
                loss = criterion(output, Y_batch)
                _, prediction = torch.max(output, 1)
                

                batches_val_loss.append(loss.item())
                batches_val_prec.append((Y_batch == prediction).to("cpu").sum() / len(Y_batch))

        print(f"""
                Epoch : {i+1}/{EPOCHS}

                    Train loss: {np.mean(batches_train_loss)}
                    Val loss: {np.mean(batches_val_loss)}
                    Val precision : {np.mean(batches_val_prec)}
        """)

        scheduler.step(np.mean(batches_val_prec))


