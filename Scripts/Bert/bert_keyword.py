"""Extract Keywords Using Bert"""

import pandas as pd
from tqdm import tqdm
from keybert import KeyBERT


def extract_keywords(text: str | list[str], threshold=0.5):
    model = KeyBERT()

    if isinstance(text, str):
        tqdm_disable = True
        texts = [text]
    else:
        texts = text
        tqdm_disable = False

    out_texts = []
    for text in tqdm(texts, disable=tqdm_disable):
        keywords = model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            use_mmr=True,
            diversity=0.8,
            top_n=3,
        )
        out_texts.append(
            [
                x[0]
                for x in
                keywords[:1] + list(filter(lambda x: x[-1] > 0.5, keywords[1:]))
            ]
        )

    return out_texts

if __name__ == "__main__":
    import re

    df = pd.read_csv("../../Data/beige_books.csv")
    texts = df.text.tolist()
    texts = [re.sub("(\\\\r)|(\\\\n)|(\\\\u)|(\\\\)", "", text) for text in texts]

    keywords = extract_keywords(texts)

    df['keywords'] = keywords
    df = df[['date', 'district', 'keywords']]
    df.to_csv("../Data/beige_books_kws.csv", index=False)
