import pickle
import pandas as pd

from Scripts.Viz.metrics import get_scores
from Scripts.Viz.plots import accuracy_plot, correlation_plot, topic_plot, shap_plot

def load():

    docs = pd.read_parquet('../Data/data/test_statements.parquet')

    prices = pd.read_parquet('../Data/data/prices.parquet')
    prices = prices.loc[docs.index]
    
    with open('./models/tfidf.pkl', 'rb') as f:
        tfidf_model = pickle.load(f)

    with open('./models/transformer.pkl', 'rb') as f:
        transformer_model = pickle.load(f)

    return docs, prices, tfidf_model, transformer_model


if __name__ == '__main__':

    
    docs, prices, tfidf_model, transformer_model = load()

    print(get_scores(prices, tfidf_model.predict(docs)))
    print(get_scores(prices, transformer_model.predict(docs)))

    accuracy_plot(docs, prices, {'tfidf':tfidf_model, 'transformer':transformer_model})

    correlation_plot(docs, prices, {'tfidf':tfidf_model, 'transformer':transformer_model})

    topic_plot(*tfidf_model[1:], prices)
    
    examples = [
        '2012-12-12',
        '2021-06-16',
        '2009-08-12',
        '2019-03-20',
        '2010-05-09',
    ]

    shap_plot(transformer_model, docs, prices, examples, 30, '\s')