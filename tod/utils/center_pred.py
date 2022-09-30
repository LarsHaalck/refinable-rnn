import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def test_center_pred(dataset, criterion):
    loader = DataLoader(dataset, batch_size=1)

    avg_loss = torch.tensor(0)
    for _, label in tqdm(loader):
        output = torch.tensor([[0.0, 0.0]])
        loss = criterion(output, label)
        avg_loss += loss / len(loader)

    return avg_loss.to("cpu")
