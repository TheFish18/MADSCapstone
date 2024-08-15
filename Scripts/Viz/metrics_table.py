"""prints a table of metrics that can be embedded into markdown."""

import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score


df_train = pd.read_csv("../../Data/Models/Second/train_88.csv")
df_val = pd.read_csv("../../Data/Models/Second/val_88.csv")
df_test = pd.read_csv("../../Data/Models/Second/test_88.csv")

dfs_names = [(df_train, 'train'), (df_val, 'val'), (df_test, 'test')]

metrics = [
    (r2_score, "R2"),
    (mean_squared_error, "MSE"),
    (root_mean_squared_error, "RMSE"),
]
n = len(metrics) + 1
metric_fns, metric_names = zip(*metrics)

header = r"| Dataset | " + r" | ".join(metric_names) + r" |"
splitter = r"| ----- " * n + r"|"

print(header)
print(splitter)

for df, ds_name in dfs_names:
    metrics = [metric_fn(df['y_true'], df['y_pred']) for metric_fn in metric_fns]
    metrics = [f"{metric: 0.4f}" for metric in metrics]

    print(rf"| {ds_name} | " + r" | ".join(metrics) + r" |")
