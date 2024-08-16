"""Train Bert Regressor on Beige Books."""
import os.path
from functools import partial
from typing import Any, Callable, Dict, List

from tqdm import tqdm

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from Scripts.Bert.bert_regressor import BertRegressor


def _accum_metrics(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        metrics: Dict[str, List[float]],
        metric_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        **kwargs
) -> Dict[str, List[float]]:
    for metric, val in kwargs.items():
        metrics[metric].append(val)

    for metric, fn in metric_fns.items():
        val = fn(y_pred, y_true)
        metrics[metric].append(val.item())

    return metrics


def _avg_metrics(
        metrics: Dict[str, List[float]],
) -> Dict[str, float]:
    return {name: sum(vals) / len(vals) for name, vals in metrics.items()}


def _print_and_log(s: str, save_p):
    with open(save_p, 'a+') as f:
        f.write(s + '\n')

    print(s)


def train(
        model: BertRegressor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: Any,
        epochs: int,
        model_name: str,
        save_directory: str,
        print_freq=10
):
    model = model.to(device)
    metric_fns = {'r2': r2_score}
    accum_metrics = partial(_accum_metrics, metric_fns=metric_fns)
    save_dir = os.path.join(save_directory, model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_txt = os.path.join(save_dir, 'log.txt')

    for epoch in range(0, epochs + 1):
        # training
        train_metrics = {name: [] for name in metric_fns}
        train_metrics['loss'] = []

        train_n = len(train_loader)
        for i, batch in tqdm(enumerate(train_loader), total=train_n):
            i += 1
            if i > 1: break
            model.train()
            optimizer.zero_grad()

            y_pred = []
            y_true = []
            for X, y in batch:
                y_pred.append(model(X))
                y_true.append(y[None, :])

            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)

            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()

            train_metrics = accum_metrics(y_pred, y_true, train_metrics, loss=loss.item())

            if (i % print_freq) == 0:
                tmp = {name: vals[-1] for name, vals in train_metrics.items()}
                _print_and_log(f'\ttrain batch: {i}, {tmp}', save_txt)

        # validation
        val_n = len(val_loader)
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader), total=val_n):
                i += 1
                if i > 1: break
                y_true = []
                y_pred = []
                for X, y in batch:
                    y_pred.append(model(X))
                    y_true.append(y[None, :])

                y_true = torch.concat(y_true)
                y_pred = torch.concat(y_pred)

                loss = loss_fn(y_pred, y_true)

                val_metrics = accum_metrics(y_pred, y_true, val_metrics, loss=loss.item())

                if (i % print_freq) == 0:
                    tmp = {name: vals[-1] for name, vals in val_metrics.items()}
                    _print_and_log(f'\tvalidation batch: {i}, {tmp}', save_txt)

        _print_and_log(
            f"epoch: {epoch}, train_metrics: {_avg_metrics(train_metrics)}, test_metrics: {_avg_metrics(val_metrics)}",
            save_txt
        )

        torch.save(model.state_dict(), os.path.join(save_dir, f"bert_regressor_{epoch}.pt"))


if __name__ == "__main__":
    from argparse import ArgumentParser

    import torch.nn as nn
    from torch.optim import Adam

    from Scripts.Bert.bert_regressor import BertRegressor
    from Scripts.Dataset.fomc_datasets import FOMCImpactDataset, to_dataloader, train_val_test_split

    parser = ArgumentParser(description="Train a BertRegressor on Beige Books")
    parser.add_argument("--model_name", "-n", type=str, default="BertRegressor", help="name of the model, default 'BertRegressor'")
    parser.add_argument("--device", "-d", type=str, default='cpu', help='device to use when training model, default cpu')
    args = parser.parse_args()

    p_bb = "Data/beige_books.csv"
    p_fomc = "Data/fomc_impact.csv"

    save_dir = "Dataset/Models/BERTModels"
    model_name = args.model_name
    device = args.device
    epochs = 100
    lr = 1e-3
    batch_size = 10

    dset = FOMCImpactDataset(p_bb, p_fomc, device=device)
    train_loader, val_loader, test_loader = to_dataloader(
        *train_val_test_split(fomc_dset=dset),
        dataloader_kwargs={'batch_size': batch_size, 'shuffle': False}
    )

    model = BertRegressor()
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        epochs,
        model_name=model_name,
        save_directory=save_dir,
        print_freq=1
    )
