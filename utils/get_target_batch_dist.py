import pandas as pd
from torch import dist

def get_target_batch_dist(Y_batch):
    dists = {}
    for k,v in pd.Series(Y_batch.cpu()).value_counts().items():
        dists[k] = v/len(Y_batch)*100
    return dists
