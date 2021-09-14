import numpy as np
import torch.utils.data as data


class WeightedSampledDataset(data.Dataset):
    def __init__(self, datasets: list, ratios: list):
        """
        Args:
            datasets (list):
            ratios (list):
        """
        self._dataset_list = datasets
        self._sample_ratio_list = ratios
        self._group = self.build_group()
        self._group_keys = list(self._group.keys())
        self._group_lengths = [int(len(g) * r) for (gn, g), r in zip(self._group.items(), ratios)]
        self._indices = np.cumsum(self._group_lengths)

    def build_group(self):
        group = {}
        for data in self._dataset_list:
            if data['data_name'] not in group:
                group[data['data_name']] = [data]
            else:
                group[data['data_name']].append(data)

        return group

    def idx2gid(self, idx):
        _gid = 0
        for i in self._indices:
            if idx < i:
                break
            _gid += 1
        return _gid

    def __len__(self):
        return sum(self._group_lengths)

    def __getitem__(self, idx):
        # idx to group_id
        gid = self.idx2gid(idx % len(self))

        # random sampling
        group = self._group[self._group_keys[gid]]
        return group[np.random.randint(len(group))]
