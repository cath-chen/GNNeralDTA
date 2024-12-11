import copy

import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from create_data import create_dataloader
from models import AttentionGNNeral
from utils import *


# TODO: k-fold

def train(model, train_loader, device, learn_rate=0.1, epochs=100, n_splits=1):
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=learn_rate)

    loss_fn = nn.MSELoss()

    model.to(device)
    best_loss = 2 ** 16
    best_epoch = 0
    best_model = model

    for epoch in (pbar := tqdm.tqdm(range(epochs), total=epochs, unit='epochs', leave=False)):

        total_loss = 0
        count = 0

        for drug, prot, y in (pbar2 := tqdm.tqdm(train_loader, total=len(train_loader), unit='batches', leave=False)):
            opt.zero_grad()

            drug = drug.to(device)
            prot = prot.to(device)
            y = y.view(-1, 1).to(device)

            pred = model(drug, prot)

            loss = loss_fn(pred, y)
            loss.backward()

            total_loss += loss.item()
            count += 1

            pbar2.set_description(f'loss={loss.item():10.5f} rmse={rmse(pred, y):10.5f}')

        avg_loss = total_loss / count

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        pbar.set_description(f'avg_loss={total_loss / count:10.5f} best_loss={best_loss:10.5f} best_epoch={best_epoch:4}')

    return best_model


def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    truth = []
    with torch.no_grad():
        for drug, target, y in dataloader:
            drug = drug.to(device)
            target = target.to(device)

            preds += model(drug, target).tolist()
            truth += y.tolist()

    loss = F.l1_loss(torch.tensor(preds), torch.tensor(truth))

    return loss, preds


if __name__ == '__main__':
    train_loader, test_loader = create_dataloader(batch_size=64)

    for drugs, prots, y in train_loader:
        drug_dim = drugs.x.shape[1]
        prot_dim = prots.x.shape[1]
        break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AttentionGNNeral(drug_dim, prot_dim, 50, time=False, attention='linear')

    train(model, train_loader, device)
