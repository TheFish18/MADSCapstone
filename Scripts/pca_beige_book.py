import matplotlib.pyplot as plt
import pandas as pd
from typing import Union
from pathlib import Path

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from Scripts.preprocess_data import merge_beige_books_impact

PathLike = Union[str, Path]

def pca(p_beige_books: str | Path, p_fomc_impact: str | Path, vectorizer: TfidfVectorizer):
    """
    5 components explained about 10% of variance
    100 components explained about 30% of variance
    1000 components explained
    Args:
        p_beige_books:
        p_fomc_impact:
        vectorizer:

    Returns:

    """
    df_beige_book = pd.read_csv(p_beige_books)
    df_beige_book.date = pd.to_datetime(df_beige_book.date)

    df_fomc_impact = pd.read_csv(p_fomc_impact)
    df_fomc_impact.date = pd.to_datetime(df_fomc_impact.date)
    df_fomc_impact = df_fomc_impact.dropna()

    df = merge_beige_books_impact(df_beige_book, df_fomc_impact)

    text_vec = vectorizer.transform(df.text)
    pca = PCA(3)
    x = pca.fit_transform(text_vec)

    x1, x2 = x[:, 0], x[:, 1]

    palette = sns.color_palette("icefire", as_cmap=True)
    sns.scatterplot(x=x1, y=x2, hue=df.diff_norm, palette=palette)
    plt.show()

if __name__ == "__main__":
    import pickle
    from pathlib import Path

    p_vec = Path("../Data/tfidf_vectorizer.pkl")
    p_bb = Path("../Data/beige_books.csv")
    p_impact = Path("../Data/fomc_impact.csv")

    with p_vec.open('rb') as f:
        vec = pickle.load(f)

    pca(p_bb, p_impact, vec)
