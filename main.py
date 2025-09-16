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
