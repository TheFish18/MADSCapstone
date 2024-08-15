r"""
Bert Regressor.

Bert portion of model is fine-tuned on economic/finance language, and outputs sentiment scores.
Regressor portion of model, takes the sentiment scores and outputs the predicted output on the market.

The output of the Bert model is a (13, 3).
The rows (13) correspond to the 12 Districts + 1 national summary.
The Columns relate to the sentiment scores.
The Regressor uses two convolution blocks and one FNN:
    - convolution block with kernel size (13, 1)
    - convolution block with kernel size(1, 3)
    - FNN
The purpose of the two convolution blocks is to
consolidate information across one district
and all sentiments and one sentiment and all district.
Matrix multiplication is then used to recreate a (13, 3)
tensor of (district, sentiment)
    - conv2d(13, 1) outputs (1, 3)
    - conv2d(1, 3) outputs (13, 1)
    - $(13, 1) = (13, 1) \cdot (1, 3)$
The sentiment scores, coming from Bert, are also run through a
FNN which are then added to the output of the matrix multiplication

Examples
--------
from datasets import FOMCImpactDataset

p_bb = "../Data/beige_books.csv"
p_fomc = "../Data/fomc_impact.csv"

dset = FOMCImpactDataset(p_bb, p_fomc, device='mps')
x, y = dset[2]

model = BertRegressor().to('mps')

y_pred = model(x)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification


class BertRegressor(nn.Module):
    """Bert Regressor"""

    def __init__(self, start_channels=32):
        """Init Bert Regressor."""
        super().__init__()

        self.bert = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.bert.requires_grad_(False)

        self.district_block = nn.Sequential(  # focuses on district
            nn.Conv2d(1, start_channels, kernel_size=(13, 1)), nn.Tanh()
        )
        self.score_block = nn.Sequential(  # focuses on sentiment score
            nn.Conv2d(1, start_channels, kernel_size=(1, 3)), nn.Tanh()
        )

        self.expand_block = nn.Sequential(nn.Conv2d(1, start_channels, kernel_size=1), nn.Tanh())

        self.linear_layers = nn.Sequential(
            nn.Linear(start_channels * 2 * 13 * 3, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.final_conv_2d = nn.Conv2d(start_channels * 2, start_channels, (13, 3))
        self.final_conv_1d = nn.Conv1d(1, 1, start_channels)

    def forward(self, x):
        """Compute y from x."""
        x = torch.concat([self.bert(**inputs)[0].mean(dim=0, keepdim=True) for inputs in x])
        x = x[None, None, :, :]
        xh = self.district_block(x)
        xw = self.score_block(x)

        xhw = xw @ xh
        x = self.expand_block(x)
        x = torch.concat((x, xhw), dim=1)

        linear_x = self.linear_layers(x.flatten(1))

        x = self.final_conv_2d(x)
        x = F.tanh(x)
        x = self.final_conv_1d(x[:, :, 0, 0])
        x = linear_x + x
        x = F.tanh(x) / 10

        return x
