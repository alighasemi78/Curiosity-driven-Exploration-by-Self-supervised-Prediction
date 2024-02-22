from torch.utils.data import Dataset


class NonSequentialDataset(Dataset):
    def __init__(self, *arrays):
        super().__init__()
        self.arrays = [array.reshape(-1, *array.shape[2:]) for array in arrays]

    def __getitem__(self, index):
        return [array[index] for array in self.arrays]

    def __len__(self):
        return len(self.arrays[0])
