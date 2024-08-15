from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from Scripts.Bert.bert_regressor import BertRegressor
from Scripts.Data.fomc_datasets import FOMCImpactDataset, train_val_test_split


def main(model_path: Path, device='mps'):
    p_bb = "../Data/beige_books.csv"
    p_fomc = "../Data/fomc_impact.csv"

    dset = FOMCImpactDataset(p_bb, p_fomc, device=device)
    train_dset, val_dset, test_dset = train_val_test_split(dset)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    model = BertRegressor()
    model.load_state_dict(state_dict)
    model.to(device)

    n = len(test_dset)
    rows = []
    for i, (X, y) in tqdm(test_dset, total=n):
        y_pred = model(X)
        rows.append({'y_pred': y_pred.item(), 'y_true': y.item()})

    df = pd.DataFrame.from_records(rows)
    return df


if __name__ == "__main__":
    model_path = Path("../Data/Models/Second/bert_regressor_50.pt")
    df = main(model_path)
    p = "../Data/Models/Second/test_50.csv"
    df.to_csv(p, index=False)
