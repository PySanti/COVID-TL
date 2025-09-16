# COVID-TL

El objetivo de este proyecto sera utilizar un modelo basado en cnn (convolutional neural network) aplicando transfer learning para la deteccion de COVID-19 en imagenes de rayos X de pechos de pacientes con y sin COVID-19.

El [dataset](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2) a utilizar contiene ~85k imagenes de rayos x de pechos de pacientes con y sin covid-19.

# Informacion de dataset:

Segun el link dispuesto, tenemos la siguiente informacion del dataset.

![Distribucion de data](./images/data_dist.png)

Las imagenes son del siguiente tipo:

![title](./images/muestra_1.png)
![title](./images/muestra_2.png)
![title](./images/muestra_3.png)

Es importante destacar que *las imagenes no tienen una resolucion estandar*. Lo anterior implica agregar tecnicas de Resizing previo al entrenamiento.



# Carga de rutas de imagenes

A traves de la siguiente funcion se crearon los datasets que almacenaran las rutas de las imagenes que se usaran para entrenamiento:

```python

import os

def load_dataset(type_, data_path):
    """
        Funcion creada para generar los paths_sets
        de las imagenes teniendo en cuenta la estructura
        de almacenamiento de las imagenes:
        o

    {data_path} 
    ├───test
    ├───train
    └───val
    ├─── test.txt
    ├─── val.txt
    ├─── test.txt
        
    """
    if type_ not in ['train', 'test', 'val']:
        raise Exception("Parametro con valor invalido !")

    file_path = os.path.join(data_path, f"{type_}.txt")
    X = []
    Y = []

    with open(file_path, 'r') as file_:
        for i, line in enumerate(file_):
            file_path = os.path.join(data_path, f"{type_}/{line.split(' ')[1]}")
            target = 1 if line.split(" ")[2] == 'positive' else 0
            X.append(file_path)
            Y.append(target)

    return X, Y

```
# Obtencion de metricas para normalizar

Para poder normalizar un conjunto de datos (necesario para el entrenamiento de redes neuronales)
se deben encontrar dos metricas: media y desviacion estandar.

Estas dos metricas se obtienen del conjunto de imagenes de entrenamiento por cada canal.

Para lograrlo, utilizamos la siguiente funcion:

```python

import numpy as np
from torch.utils.data import DataLoader

def normalization_metrics_calc(dataset):
    loader = DataLoader(dataset=dataset, batch_size=1, num_workers=8)

    n_pixels_total = 0
    sum_rgb = np.zeros(3, dtype=np.float64)
    sum_sq_rgb = np.zeros(3, dtype=np.float64)

    for i, image in enumerate(loader):
        print(f"{i}/{len(dataset)}", end="\r")
        image = image[0]
        array = image.squeeze(0).numpy().astype(np.float32) / 255.0  # Escalar a [0,1]
        h, w, _ = array.shape
        n_pixels = h * w
        n_pixels_total += n_pixels

        # Suma de valores y suma de cuadrados por canal
        sum_rgb += array.reshape(-1, 3).sum(axis=0)
        sum_sq_rgb += (array.reshape(-1, 3) ** 2).sum(axis=0)

    mean = sum_rgb / n_pixels_total
    std = np.sqrt(sum_sq_rgb / n_pixels_total - mean ** 2)

    return mean.tolist(), std.tolist()
```

Esta funcion toma un dataset que wrappea la lista de rutas de imagenes y lo recorre, dicho dataset es el siguiente:

```python

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

```

En el `main.py`:

```python

import kagglehub
from utils.NormDataset import NormDataset
from utils.load_dataset import load_dataset
from utils.normalization_metrics_calc import normalization_metrics_calc

if __name__ == "__main__":

    data_path = kagglehub.dataset_download("andyczhao/covidx-cxr2")
    train_X, train_Y = load_dataset('train', data_path)
    val_X, val_Y = load_dataset('val', data_path)
    test_X, test_Y = load_dataset('test', data_path)

    trainset = NormDataset(train_X)
    print(normalization_metrics_calc(trainset))
```

Encontramos los siguientes datos:

```

media : [0.5162912460480391, 0.5162851561852722, 0.5162961087834854]

std : [0.24806341486746794, 0.24806204203020304, 0.248068391383844])
```

# Esqueleto de entrenamiento


