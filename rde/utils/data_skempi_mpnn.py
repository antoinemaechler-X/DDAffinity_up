import math
import torch
from torch.utils.data._utils.collate import default_collate


DEFAULT_PAD_VALUES = {
    'aa': 21,
    'aa_masked': 21,
    'aa_true': 21,
    'chain_nb': -1, 
    'pos14': 0.0,
    'chain_id': ' ', 
    'icode': ' ',
}

class PaddingCollate(object):

    def __init__(self, patch_size, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, eight=True):
        super().__init__()
        self.patch_size = patch_size
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _filter_and_get_keys(data_list):
        # Filter out samples missing centrality and collect keys
        filtered_data = []
        missing_centrality = []
        
        for data in data_list:
            complex_id = data["wt"].get('complex', 'unknown')
            if 'centrality' not in data["wt"]:
                missing_centrality.append(complex_id)
                continue
            filtered_data.append(data)
        
        if missing_centrality:
            print(f"Warning: Filtered out {len(missing_centrality)} samples missing centrality: {missing_centrality}")
        
        if not filtered_data:
            raise ValueError("No samples with centrality found in batch!")
        
        # Get intersection of keys from filtered data
        keys = set(filtered_data[0]["wt"].keys())
        for d in filtered_data[1:]:
            keys = keys.intersection(d["wt"].keys())
        
        return filtered_data, keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        # Filter out samples missing centrality and get common keys
        filtered_data, keys = self._filter_and_get_keys(data_list)
        
        if len(filtered_data) == 0:
            raise ValueError("No valid samples in batch after filtering!")
        
        max_length = max([data["wt"][self.length_ref_key].size(0) for data in filtered_data])
        if max_length < self.patch_size:
            max_length = self.patch_size
        
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8

        data_list_padded = []
        for data in filtered_data:
            data_dict = {}
            for flag in ["wt", "mt"]:
                data_padded = {
                    k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                    for k, v in data[flag].items()
                    if k in keys
                }
                data_padded['mask'] = self._get_pad_mask(data[flag][self.length_ref_key].size(0), max_length)
                data_dict[flag] = data_padded
            data_list_padded.append(data_dict)

        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)
        ddG = torch.tensor([data["wt"]["ddG"] for data in filtered_data], dtype=torch.float32).unsqueeze(-1)
        batch['ddG'] = ddG

        return batch

