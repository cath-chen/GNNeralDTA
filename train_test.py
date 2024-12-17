import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch_geometric.nn as gnn
import tqdm
from lifelines.utils import concordance_index as ci
from sklearn.metrics import mean_squared_error as mse
from torch import nn

from create_data import create_dataloader
from models import AttentionGNNeral


def train(model, train_loader, device, learn_rate=0.0005, epochs=100, val_loader=None, early_stop_epochs=100):
    start = time.time()

    opt = torch.optim.Adam(model.parameters(), lr=learn_rate)

    loss_fn = nn.MSELoss()

    model.to(device)
    best_mse = 2 ** 16
    best_ci = 2 ** 16
    best_loss = 2 ** 16
    best_epoch = 0
    best_model = model.state_dict()

    for epoch in (pbar := tqdm.tqdm(range(epochs), total=epochs, unit='epochs', leave=False)):
        model.train()

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
            opt.step()

            total_loss += loss.item()
            count += 1

            pbar2.set_description(f'mse={loss.item():6.3f} avg_mse={total_loss / count:6.3f}')

        loss = total_loss / count

        if val_loader is not None:
            ci_score, mse_score, _, _ = evaluate(model, val_loader, device)

            if mse_score < best_mse:
                best_mse = mse_score
                best_ci = ci_score
                best_epoch = epoch
                best_model = model.state_dict()
                best_loss = loss

            pbar.set_description(
                f'loss={loss:6.3f} test_mse={mse_score:6.3f} best=[epoch={best_epoch:3} loss={best_loss:6.3f} mse={best_mse:6.3f} ci={best_ci:6.3f}]')

        else:
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                best_model = model.state_dict()

            pbar.set_description(f'loss={loss:6.3f} best=[epoch={best_epoch + 1:3} loss={best_loss:6.3f}]')

        if early_stop_epochs and epoch - best_epoch >= early_stop_epochs:
            print(f"stopping early after {epoch} epochs")
            break

    end = time.time()

    model.load_state_dict(best_model)

    train_ci, train_mse, _, _ = evaluate(model, train_loader, device)
    results = {'runtime': end - start, 'train_ci': train_ci, 'train_mse': train_mse, 'best_epoch': best_epoch + 1}
    if val_loader is not None:
        test_ci, test_mse, _, _ = evaluate(model, val_loader, device)
        results['test_ci'] = test_ci
        results['test_mse'] = test_mse

    return model, results


def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)
    preds = []
    truth = []
    with torch.no_grad():
        for drug, target, y in dataloader:
            drug = drug.to(device)
            target = target.to(device)
            y = y.view(-1, 1).to(device)

            preds += model(drug, target).detach().cpu().tolist()
            truth += y.detach().cpu().tolist()

    y = np.array(truth)
    f = np.array(preds)

    ci_score = ci(y, f)
    mse_score = mse(y, f)

    return ci_score, mse_score, y, f


def append_print(filename, text):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a') as f:
        f.write(text + '\n')
    print(text)


# use this, more efficient:
def smart_tune(drug_dim, prot_dim, train_loader, val_loader, device, epochs=100, test_attention=True):
    filename = f"tune/{time.strftime('%Y%m%d-%H%M%S')}.txt"

    if test_attention:
        config = {}

        for config['attention'] in ['linear', 'cross', 'reduced-cross']:
            model = AttentionGNNeral(drug_dim, prot_dim, **config)
            _, results = train(model, train_loader, device,
                               val_loader=val_loader, early_stop_epochs=epochs // 3, epochs=epochs)
            append_print(filename, str(config))
            append_print(filename, str(results))

        return

    params = {'learn_rate': [0.001, 0.0001, 0.00001], 'attention_dim': [64, 128, 256],
              'conv': [gnn.GCNConv, gnn.SAGEConv, gnn.GraphConv, gnn.GATConv],
              'prot_gnn_layers': [2, 4, 6], 'drug_gnn_layers': [3, 5, 7],
              'gnn_dropout': [0.0, 0.1, 0.2], 'fnn_dropout': [0.0, 0.1, 0.2, 0.33, 0.5]}
    config = {'learn_rate': 0.001}
    prev_config = {}
    count = 0
    append_print(filename, str(device))
    while config != prev_config and count < 1:  # stop once the model is not changing  anymore
        count += 1
        prev_config = config.copy()
        for key in params:
            best_mse = 2 ** 16
            best_value = params[key][0]
            for config[key] in params[key]:
                append_print(filename, str(config))
                model = AttentionGNNeral(drug_dim, prot_dim, **config)
                _, results = train(model, train_loader, device, learn_rate=config['learn_rate'],
                                   val_loader=val_loader, early_stop_epochs=epochs // 3, epochs=epochs)
                results = {key: round(value, 3) for key, value in results.items()}
                append_print(filename, str(results))
                if results['test_mse'] < best_mse:
                    best_mse = results['test_mse']
                    best_value = config[key]
            config[key] = best_value

    config['attention'] = 'cross'
    best_mse = 2 ** 16
    best_value = 1
    for config['num_heads'] in [16, 64, 256]:
        append_print(filename, str(config))
        model = AttentionGNNeral(drug_dim, prot_dim, **config)
        _, results = train(model, train_loader, device, learn_rate=config['learn_rate'],
                           val_loader=val_loader, early_stop_epochs=epochs // 3, epochs=epochs)
        results = {key: round(value, 3) for key, value in results.items()}
        append_print(filename, str(results))
        if results['test_mse'] < best_mse:
            best_mse = results['test_mse']
            best_value = config['num_heads']
    config['num_heads'] = best_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Attention! GNNeral")
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batchsize', type=int, default=128)
    parser.add_argument('-t', '--tune', action='store_true')
    parser.add_argument('-f', '--folds', type=int, default=1)
    parser.add_argument('-l', '--load', type=str, default=None)
    parser.add_argument('-a', '--attention', type=str, default='linear')
    args = parser.parse_args()

    if args.tune or args.load is not None:
        args.folds = 1

    train_loader, test_loader = create_dataloader(batch_size=args.batchsize, n_splits=args.folds)

    for drugs, prots, y in test_loader:
        drug_dim = drugs.x.shape[1]
        prot_dim = prots.x.shape[1]
        break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.tune:
        smart_tune(drug_dim, prot_dim, train_loader, test_loader, device, args.epochs, test_attention=True)

    elif args.load is not None:
        model = AttentionGNNeral(drug_dim, prot_dim, attention=args.attention)

        with open(args.load, 'rb') as f:
            model_dict = pickle.load(f)

        model.load_state_dict(model_dict)

        train_ci, train_mse, _, _ = evaluate(model, train_loader, device)
        test_ci, test_mse, _, _ = evaluate(model, test_loader, device)

        print(f"{train_ci=:.4f} {train_mse=:.4f}")
        print(f"{test_ci=:.4f} {test_mse=:.4f}")


    elif args.folds == 1:
        model = AttentionGNNeral(drug_dim, prot_dim, attention=args.attention)

        model, results = train(model, train_loader, device, epochs=args.epochs)

        print("train results:")
        print(results)

        ci_score, mse_score, _, _ = evaluate(model, test_loader, device)
        print(f"test ci score: {ci_score} test mse: {mse_score}")

    else:
        filename = f"train/{time.strftime('%Y%m%d-%H%M%S')}"

        model = AttentionGNNeral(drug_dim, prot_dim, attention=args.attention)
        append_print(filename + '.txt', args.attention)

        splits = train_loader

        models, ci_scores, mse_scores = [], [], []

        for i, (train_loader, val_loader) in enumerate(splits):
            model, results = train(model, train_loader, device, epochs=args.epochs, val_loader=val_loader)

            ci_score = results['train_ci']
            mse_score = results['train_mse']

            test_ci_score, test_mse_score, _, _ = evaluate(model, test_loader, device)

            models.append(model)
            ci_scores.append(ci_score)
            mse_scores.append(mse_score)

            append_print(filename + ".txt",
                         f"split={i + 1} val_ci_score={ci_score:.4f} val_mse_score={mse_score:.4f} {test_ci_score=:.4f} {test_mse_score=:.4f} best_epoch={results['best_epoch']:3} runtime={results['runtime']:7.2f}s")

        best_model = models[np.argmin(ci_scores)]

        model_dict = best_model.state_dict()

        with open(filename + "_model.pkl", 'wb') as f:
            pickle.dump(model_dict, f)
