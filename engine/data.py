import numpy as np
from .autograd import Tensor


# =============================================================================
# DATASET
# =============================================================================

class Dataset:

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TensorDataset(Dataset):

    def __init__(self, *tensors):
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].data.shape[0]

    def __getitem__(self, idx):
        return tuple(Tensor(t.data[idx]) for t in self.tensors)


# =============================================================================
# DATALOADER
# =============================================================================

class DataLoader:

    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        indices = np.random.permutation(n) if self.shuffle else np.arange(n)

        for start in range(0, n, self.batch_size):
            batch_idx = indices[start: start + self.batch_size]
            batch = self.dataset[batch_idx]
            if isinstance(batch, tuple):
                yield batch
            else:
                yield (batch,)


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    X = Tensor(np.random.randn(100, 4).astype(np.float32))
    y = Tensor(np.random.randn(100, 1).astype(np.float32))

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    print(f"dataset size : {len(ds)}")
    print(f"num batches  : {len(dl)}")

    for i, (xb, yb) in enumerate(dl):
        print(f"batch {i}  x:{xb.shape}  y:{yb.shape}")
