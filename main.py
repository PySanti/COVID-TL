from numpy.matrixlib import test
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import kagglehub
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.load_dataset import load_dataset
from utils.load_model import load_model
from utils.ImagesDataset import ImagesDataset
from torchvision import transforms
from utils.MACROS import BATCH_SIZE, EPOCHS, IMAGE_SIZE, MEANS, STDS

from utils.plot_model_performance import plot_model_performance
from utils.precision import precision

if __name__ == "__main__":

    data_path = kagglehub.dataset_download("andyczhao/covidx-cxr2")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


    train_transformer = transforms.Compose([
        # Redimensionar y recorte aleatorio para romper consistencia espacial
        transforms.Resize((IMAGE_SIZE[0], IMAGE_SIZE[1])),
        transforms.RandomResizedCrop((int(IMAGE_SIZE[0]*0.9), int(IMAGE_SIZE[1]*0.9)), scale=(0.7, 1.0), ratio=(0.75, 1.33)),


        # Augmentations agresivos
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),

        # Cutout (DropBlock espacial)
#        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),

        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS)
    ])


    val_transformer = transforms.Compose([
        transforms.Resize((IMAGE_SIZE[0], IMAGE_SIZE[1])), # redimensionar 
        transforms.CenterCrop((IMAGE_SIZE[0]*0.9, IMAGE_SIZE[1]*0.9)), # recordar desde el centro para consistencia
        

        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS)
    ])

    trainset = ImagesDataset(
            *load_dataset('train', data_path), 
            train_transformer=train_transformer, 
            val_transformer=val_transformer,
            train_dataset=True,
            minority_class=0)

    valset = ImagesDataset(
            *load_dataset('val', data_path), 
            train_transformer=train_transformer, 
            val_transformer=val_transformer,
            train_dataset=False,
            minority_class=0)

    testset = ImagesDataset(
            *load_dataset('test', data_path), 
            train_transformer=train_transformer, 
            val_transformer=val_transformer,
            train_dataset=False,
            minority_class=0)


    trainloader = DataLoader(trainset, BATCH_SIZE, shuffle=True, num_workers=5, persistent_workers=True, pin_memory=True)
    valloader = DataLoader(valset, BATCH_SIZE, shuffle=False, num_workers=5, persistent_workers=True, pin_memory=True)
    testloader = DataLoader(testset, BATCH_SIZE, shuffle=False, num_workers=5, persistent_workers=True, pin_memory=True)

    model = load_model(2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    weights = compute_class_weight('balanced', classes=np.unique(trainset.Y), y=trainset.Y)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, weight=torch.Tensor(weights).to(DEVICE))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",patience=8)



    epochs_train_loss = []
    epochs_val_loss = []

    for i in range(EPOCHS):
        t1 = time.time()

        batches_train_loss = []
        batches_val_loss = []
        batches_val_prec =[]
        batches_test_prec =[]
        
        model.train()
        for a, (X_batch, Y_batch) in enumerate(trainloader):
            print(f"\t\ttrain batch : {a}/{len(trainloader)}", end="\r")
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

            optimizer.zero_grad()

            output = model(X_batch)
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()


            batches_train_loss.append(loss.item())

        print()
        model.eval()

        with torch.no_grad():

            # validation

            for a, (X_batch, Y_batch) in enumerate(valloader):
                print(f"\t\tval batch : {a}/{len(valloader)}", end="\r")
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

                output = model(X_batch)
                loss = criterion(output, Y_batch)
                
                batches_val_loss.append(loss.item())
                batches_val_prec.append(precision(output, Y_batch))
            
            print()
            # test

            for a, (X_batch, Y_batch) in enumerate(testloader):
                print(f"\t\ttest batch : {a}/{len(testloader)}", end="\r")
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                
                output = model(X_batch)
                batches_test_prec.append(precision(output, Y_batch))

        print(f"""
                Epoch : {i+1}/{EPOCHS}

                    Train loss: {np.mean(batches_train_loss):.4f}
                    Val loss: {np.mean(batches_val_loss):.4f}
                    Val precision : {np.mean(batches_val_prec):.4f}
                    Test precision : {np.mean(batches_test_prec):.4f}
                    Time : {time.time()-t1:.4f}

                    ___________________________________________
        """)

        epochs_train_loss.append(np.mean(batches_train_loss))
        epochs_val_loss.append(np.mean(batches_val_loss))

        scheduler.step(np.mean(batches_val_prec))
    torch.save(model, "./results/se_net/se_net.pt")
    torch.save(torch.Tensor(epochs_val_loss), "./results/se_net/epochs_loss.pt")
    plot_model_performance(epochs_train_loss, epochs_val_loss)


