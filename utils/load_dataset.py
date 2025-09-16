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

    
