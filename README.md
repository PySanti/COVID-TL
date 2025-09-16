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

