import torch
from torch.utils.data import Dataset, DataLoader


class RandomNodeSamplingDataset(Dataset):
    def __init__(self, node_mask: torch.BoolTensor):
        super().__init__()
        node_mask = node_mask.cpu()  # at cpu
        node_index = torch.arange(node_mask.size(0))
        node_index = torch.masked_select(node_index, node_mask)  # [valid_size]
        self.node_index = node_index.numpy()

    def __getitem__(self, idx):
        return self.node_index[idx]

    def __len__(self):
        return len(self.node_index)


if __name__ == '__main__':
    batch_size = 5
    mask = torch.arange(50)
    mask = (mask < 25).bool()

    dataset = RandomNodeSamplingDataset(mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_data in dataloader:
        print(batch_data)