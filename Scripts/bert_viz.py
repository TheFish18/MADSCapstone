"""Various visualization related to BertRegressor"""
import json
from pathlib import Path

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from Scripts.bert_regressor import BertRegressor

DISTRICTS = [
    'Atlanta', 'Boston', 'Chicago', 'Cleveland', 'Dallas', 'Kansas City', 'Minneapolis',
    'National Summary', 'New York', 'Philadelphia', 'Richmond', 'San Francisco', 'St Louis'
]

def log_to_df():
    p = Path("/Users/joshfisher/PycharmProjects/MADSCapstone/Data/Models/Second/log.txt")

    with p.open('r') as f:
        lines = f.readlines()


    def to_dict(line):
        line = line.replace(
            'epoch', '"epoch"'
        ).replace(
            'train_metrics', '"train_metrics"'
        ).replace(
            'test_metrics', '"test_metrics"'
        ).replace(
            "'", '"'
        )

        line = "{" + line + "}"
        data = json.loads(line)

        return data['train_metrics'], data['test_metrics']

    lines = [line for line in lines if line.startswith('e')]
    train_data, test_data = list(zip(*map(to_dict, lines)))

    df_train = pd.DataFrame.from_records(train_data)
    df_test = pd.DataFrame.from_records(test_data)

    for col in df_train.columns:
        train = df_train[col]
        test = df_test[col]

        plt.plot(train, 'g')
        plt.plot(test, 'r')
        plt.title(col)
        plt.show()

def test_ytrue_ypred_dists():
    p = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/Models/Second/test.csv"
    df = pd.read_csv(p)

    df = df.melt(value_vars=['y_true', 'y_pred'])
    print(df.head().to_string())

    sns.violinplot(df, x="value", y="variable", hue="variable")
    plt.show()


def bar_ytrue_ypred_dists():
    p = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/Models/Second/train.csv"
    df = pd.read_csv(p)

    df = df.melt(id_vars='epoch', value_vars=['y_true', 'y_pred'])
    print(df.head().to_string())

    sns.barplot(df, x="epoch", y="value", hue="variable")
    plt.show()


def bert_correct():
    p = "../Data/Models/Second/test_88.csv"
    df = pd.read_csv(p)

    pred_pos = df.y_pred > 0
    actual_pos = df.y_true > 0

    pred_neg = df.y_pred < 0
    actual_neg = df.y_true < 0

    TPs = pred_pos & actual_pos
    FPs = pred_pos & actual_neg
    FNs = pred_neg & actual_pos
    TNs = pred_neg & actual_neg

    precision = sum(TPs) / (sum(TPs) + sum(FPs))
    recall = sum(TPs) / (sum(TPs) + sum(FNs))
    iou = sum(TPs) / (sum(TPs) + sum(FPs) + sum(FNs))

    print(precision, recall, iou)
    print(TPs.mean())

    cm = confusion_matrix(pred_pos, actual_pos)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()

def scatter():
    p = "../Data/Models/Second/test_88.csv"
    df = pd.read_csv(p)

    sns.scatterplot(df, x='y_pred', y='y_true')
    plt.show()

def param_heatmap():
    model_path = r"../Data/Models/Second/bert_regressor_88.pt"
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model = BertRegressor()
    model.load_state_dict(state_dict)

    score_block = model.get_submodule("score_block")
    district_block = model.get_submodule("district_block")

    score_params = score_block.get_parameter('0.weight')
    district_params = district_block.get_parameter('0.weight')
    score_bias = score_block.get_parameter('0.bias')
    district_bias = district_block.get_parameter("0.bias")

    score_params = score_params + score_bias[:, None, None, None]
    district_params = district_params + district_bias[:, None, None, None]

    params = district_params @ score_params
    params = torch.mean(params, dim=0)[0].detach().numpy()

    df = pd.DataFrame(params, index=DISTRICTS, columns=['Positive', 'Neutral', 'Negative'])

    sns.heatmap(df)
    plt.tight_layout()
    plt.savefig("../main/embeddings/param_heatmap.png")
    plt.show()

if __name__ == "__main__":
    param_heatmap()
