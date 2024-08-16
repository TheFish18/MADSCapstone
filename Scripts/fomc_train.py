import os
import argparse
import pickle
from sklearn.decomposition import PCA, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

from Scripts.Scrape.get_statements import get_statements
from Scripts.Scrape.get_prices import get_prices
from Scripts.Data.sklearn_classes import EncoderTransformer, SentenceSelector, Condition, Examples, Splitter
from Scripts.Data.st_cache import SENTENCE_TRANSFORMER_CACHE


parser = argparse.ArgumentParser('Fit Markepulse Models')
parser.add_argument('start', help='first date')
parser.add_argument('end', help='last date')
parser.add_argument('--seed', help='random seed', default=0)
parser.add_argument('--model', help='SentenceTransformer model name', default='philschmid/bge-base-financial-matryoshka')
parser.add_argument('--save', help='save datasets', default=True)

def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


if __name__ == '__main__':
    from pathlib import Path
    save_dir = Path("Data")
    save_dir.mkdir(exist_ok=True)

    args = parser.parse_args()

    docs = get_statements(args.start, args.end)

    prices = get_prices(40, 20, docs.index.min(), docs.index.max())

    selector = SentenceSelector(
        Splitter(), 
        SENTENCE_TRANSFORMER_CACHE(args.model), 
        Condition(),
        Examples(), 
        KNeighborsRegressor(5)
    )

    tfidf_model = make_pipeline(
        selector,
        TfidfVectorizer(min_df=.05, max_df=.5),
        NMF(6, max_iter=1000, random_state=args.seed),
        LinearRegression()
    )

    transformer_model = make_pipeline(
        selector,
        EncoderTransformer(args.model),
        PCA(24),
        LinearRegression()
    )

    y = prices.loc[docs.index]

    X_train, X_test, y_train, y_test = train_test_split(docs, y, test_size=.4, random_state=args.seed)

    if args.save:
        X_train.to_parquet(save_dir / 'train_statements.parquet')
        X_test.to_parquet(save_dir / 'test_statements.parquet')
        prices.to_parquet('Data/prices.parquet')

    tfidf_model.fit(X_train, y_train)

    transformer_model.fit(X_train, y_train)

    models_dir = save_dir / "Models/FOMCModels"
    tfidf_path = models_dir / "tfidf.pkl"
    transformer_path = models_dir / "transformer.pkl"

    with tfidf_path.open('wb') as f:
        pickle.dump(tfidf_model, f)

    with transformer_path.open('wb') as f:
        pickle.dump(transformer_model, f)
