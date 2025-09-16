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
