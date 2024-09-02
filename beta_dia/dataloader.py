from torch.utils.data.dataset import Dataset

try:
    profile
except:
    profile = lambda x: x

class Map_Dataset(Dataset):
    def __init__(self, maps, valid_ion_nums, labels):
        self.maps = maps
        self.valid_ion_nums = valid_ion_nums
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        maps = self.maps[idx]  # [ion_num, 13, 50]
        y = self.labels[idx]
        valid_ion_num = self.valid_ion_nums[idx]

        return (maps, valid_ion_num, y)


class Mall_Dataset(Dataset):
    def __init__(self, malls, valid_ion_nums, labels):
        self.malls = malls
        self.valid_ion_nums = valid_ion_nums
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mall = self.malls[idx]
        y = self.labels[idx]
        valid_ion_num = self.valid_ion_nums[idx]

        return (mall, valid_ion_num, y)
