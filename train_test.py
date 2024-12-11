import copy

import torch
import torch.nn.functional as F
import tqdm

from create_data import create_dataloader
from models import AttentionGNNeral


# TODO: k-fold

def train(model, train_loader, device, learn_rate=0.001, epochs=100, n_splits=1):
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.to(device)
    best_loss = 2 ** 16
    best_epoch = 0
    best_model = model

    for epoch in (pbar := tqdm.tqdm(range(epochs), total=epochs, unit='epochs', leave=False)):

        total_loss = 0
        count = 0

        for drug, prot, y in train_loader:
            opt.zero_grad()

            drug = drug.to(device)
            prot = prot.to(device)

            pred = model(drug, prot)

            loss = F.l1_loss(pred, y)
            loss.backward()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        pbar.set_description(f'avg_loss={total_loss / count:.3f} best_loss={best_loss:.3f} best_epoch={best_epoch:4}')

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
    train_loader, test_loader = create_dataloader()

    for drugs, prots, y in train_loader:
        drug_dim = drugs.x.shape[1]
        prot_dim = prots.x.shape[1]
        break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AttentionGNNeral(drug_dim, prot_dim, 50)

    train(model, train_loader, device)
