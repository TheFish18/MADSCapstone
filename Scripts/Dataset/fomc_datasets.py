"""
Various classes that subclass torch.utils.data.Dataset to be used for training
"""
import copy
from typing import Tuple, Dict, Any, Sequence

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from transformers import BertTokenizer

from Scripts.Dataset import preprocess_data as prepro


class _BertTokenizer:
    """
    Bert Tokenizer
    """
    def __init__(self, pretrained_model_name_or_path: str, device: Any):
        """
        initialize a Bert Tokenizer
        Args:
            pretrained_model_name_or_path: see transformers.BertTokenizer.from_pretrained
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.device = device

    def tokenize(self, text: str, chunk_size=512) -> Dict[str, torch.Tensor]:
        """
        Tokenize a text and chunk into chunk_size if applicable

        adapted from https://towardsdatascience.com/how-to-apply-transformers-to-any-length-of-text-a5601410af7f
        Args:
            text: text to be tokenized
            chunk_size: size of each chunk

        Returns:
            Dictionary containing tokens as pytorch tensors
        """
        out_d = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
        tokens = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')

        for name, token in tokens.items():
            token_chunks = token[0].split(chunk_size - 2)  # -2 to account for [cls] and [sep] tokens
            for chunk in token_chunks:
                if name == "input_ids":
                    # [CLS]/101 and seperator token [SEP]/102
                    chunk = torch.concat((torch.tensor([101]), chunk, torch.tensor([102])))
                elif name == "token_type_ids":
                    pass
                elif name == "attention_mask":
                    chunk = torch.concat((torch.ones(1), chunk, torch.ones(1)))
                else:
                    raise ValueError(f"unexpected name when tokenizing: {name}")

                chunk_len = chunk.size(0)
                if chunk_len < chunk_size:
                    pad_len = chunk_size - chunk_len
                    chunk = torch.concat((chunk, torch.zeros(pad_len)))

                out_d[name].append(chunk)

        out_d = {name: torch.stack(tokens).to(self.device) for name, tokens in out_d.items()}

        return {
            'input_ids': out_d['input_ids'].long(),
            'attention_mask': out_d['attention_mask'].int(),
            'token_type_ids': out_d['token_type_ids'].int()
        }


class FOMCImpactDataset(Dataset, _BertTokenizer):
    DISTRICTS = [
        'Atlanta', 'Boston', 'Chicago', 'Cleveland', 'Dallas', 'Kansas City', 'Minneapolis',
        'National Summary', 'New York', 'Philadelphia', 'Richmond', 'San Francisco', 'St Louis'
    ]
    """Beige Books and FOMC Impact on SP500 """
    def __init__(self, p_beige_books: str, p_fomc_impacts: str, device='cpu'):
        """
        Dataset to return encoded beige book and impact of FOMC on sp500
        Args:
            p_beige_books: path to beige_books.csv
            p_fomc_impacts: path to fomc_impact.csv
        """
        Dataset.__init__(self)
        _BertTokenizer.__init__(self, 'ProsusAI/finbert', device=device)

        df_bb = pd.read_csv(p_beige_books)
        df_bb.date = pd.to_datetime(df_bb.date)

        df_fomc = pd.read_csv(p_fomc_impacts)
        df_fomc.date = pd.to_datetime(df_fomc.date)

        df = prepro.merge_beige_books_impact(df_bb, df_fomc).reset_index(drop=True)

        self.dates = df["bb_date"].sort_values(ascending=False).unique()
        self.df = df

    def __len__(self):
        """ Returns length of Dataset """
        return len(self.dates)

    def __getitem__(self, item: int) -> Tuple:
        """
        Get vectorizer encoded beige book, sp500 impact due to FOMC
        Args:
            item: index

        Returns:
            - vectorizer encoded beige book, one row for each district and one column for each word
            - impact FOMC had on sp500
        """
        date = self.dates[item]
        group = self.df.loc[self.df['bb_date'] == date]
        group = group.drop_duplicates()

        districts = group.district.unique().tolist()
        if len(districts) < 13:
            add_districts = [d for d in self.DISTRICTS if d not in districts]
            rows = []
            for district in add_districts:
                tmp = group.iloc[0].to_dict()
                tmp['district'] = district
                tmp['text'] = ""
                rows.append(tmp)

            tmp = pd.DataFrame.from_records(rows)
            group = pd.concat((group, tmp))

        X = list(map(self.tokenize, group.sort_values('district').text))
        y = group.diff_norm.iloc[0]  # all diff_norm values in group are identical

        return X, torch.tensor([y], device=self.device, dtype=torch.float32)

def train_val_test_split(
        fomc_dset: FOMCImpactDataset,
        ps=(0.8, 0.1, 0.1)
) -> Tuple[FOMCImpactDataset, FOMCImpactDataset, FOMCImpactDataset]:
    """
    train val test split FOMCImpactDataset
    Args:
        fomc_dset: FOMCImpactDataset to split
        ps: (train_proportion, val_prop, test_prop), must sum to 1. Default: (0.8, 0.1, 0.1)

    Returns:
        train dataset, val dataset, test dataset

    """

    assert sum(ps) == 1., "sum of ps must equal 1"
    n = len(fomc_dset)
    ns = np.array(ps) * n
    ns = np.round(ns)
    idxs = np.cumsum(ns).astype(int)

    val_dset = copy.deepcopy(fomc_dset)
    val_dset.dates = val_dset.dates[idxs[0]: idxs[1]]
    test_dset = copy.deepcopy(fomc_dset)
    test_dset.dates = test_dset.dates[idxs[1]:]
    train_dset = fomc_dset
    train_dset.dates = train_dset.dates[:idxs[0]]

    return train_dset, val_dset, test_dset

def to_dataloader(*fomc_dset: Sequence[FOMCImpactDataset], dataloader_kwargs):
    """
    FOMCImpactDataset to Dataloader
    Args:
        *fomc_dset: FOMCImpactDatasets
        dataloader_kwargs: DataLoader kwarg

    Returns:
        dataloader for each dataset passed in

    """
    def collate_fn(batch):
        return batch

    dataloader_kwargs['collate_fn'] = collate_fn

    return (DataLoader(dset, **dataloader_kwargs) for dset in fomc_dset)


if __name__ == "__main__":
    """ Example usage: """

    p_bb = "../Data/beige_books.csv"
    p_fomc = "../Data/fomc_impact.csv"

    dset = FOMCImpactDataset(p_bb, p_fomc)
    train_loader, val_loader, test_loader = to_dataloader(
        *train_val_test_split(fomc_dset=dset),
        dataloader_kwargs={'batch_size': 4, 'shuffle': False}
    )

    train_iter = iter(train_loader)
    print(next(train_iter))
