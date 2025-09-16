import kagglehub
from utils.load_dataset import load_dataset

data_path = kagglehub.dataset_download("andyczhao/covidx-cxr2")
train_X, train_Y = load_dataset('train', data_path)
val_X, val_Y = load_dataset('val', data_path)
test_X, test_Y = load_dataset('test', data_path)


