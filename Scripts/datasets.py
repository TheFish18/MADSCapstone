import os.path
import pickle

import pandas as pd
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer

from Scripts import preprocess_data as prepro


class FOMCImpactDataset(Dataset):
    """Beige Books and FOMC Impact on SP500 """
    def __init__(self, p_beige_books: str, p_fomc_impacts: str, vectorizer: TfidfVectorizer | str = None):
        """

        Args:
            p_beige_books: path to beige_books.csv
            p_fomc_impacts: path to fomc_impact.csv
        """

        df_bb = pd.read_csv(p_beige_books)
        df_bb.date = pd.to_datetime(df_bb.date)

        df_fomc = pd.read_csv(p_fomc_impacts)
        df_fomc.date = pd.to_datetime(df_fomc.date)

        df = prepro.merge_beige_books_impact(df_bb, df_fomc).reset_index()

        if vectorizer is None:
            vectorizer = TfidfVectorizer(stop_words='english')
        elif isinstance(vectorizer, str):
            assert os.path.splitext(vectorizer)[-1] == ".pkl", "if passing vectorizer path, must be a pickle file"
            with open(vectorizer, 'rb') as f:
                vectorizer = pickle.load(f)

        if isinstance(vectorizer, TfidfVectorizer):
            self.vectorizer = vectorizer
        else:
            raise TypeError('Expected vectorizer to be a TfidfVectorizer, a path to a TfidfVectorizer, or None')

        self.X = self.vectorizer.fit_transform(df.text)
        self.dates = df["impact_date"].sort_values(ascending=False).unique().tolist()
        self.df = df.drop(columns=['text'])  # dropped to save space

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, item):
        date = self.dates[item]
        group = self.df.loc[self.df['impact_date'] == date]
        index = group.district.sort_values().index
        X = self.X[index]
        y = group.diff_norm.iloc[0]  # all diff_norm values in group are identical
        return X, y

if __name__ == "__main__":
    p_bb = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/beige_books.csv"
    p_fomc = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/fomc_impact.csv"
    p_vec = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/tfidf_vectorizer.pkl"

    dset = FOMCImpactDataset(p_bb, p_fomc, p_vec)
    x = dset[2]
