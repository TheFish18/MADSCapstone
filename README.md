# MADSCapstone
Capstone Project For Masters in Applied Data Science - University of Michigan

# Setup
1. Install `python=3.11.9` using anaconda or other virtual environment
2. Install packages specified in `requirements.txt` using `pip install -r requirements.txt` using virtual environment
3. Install `PyTorch==2.2.2` by following their [install guide](https://pytorch.org/get-started/locally/)
4. In your repo/project root, create a folder named Data such that your project directory looks like:
   - `MADSCapstone/`
     - `Data/`
     - `Scripts/`

# Data:
Data supporting this report are openly available.

Scraping Data is challenging but even more challenging is writing fully reproducible scraping code due to the
high frequency at which webpages change, the slight inconsistencies between endpoints of one webpage to another, etc.
Therefore, in addition to the code that was used to scrape the websites, we have also included a 
[link](https://www.kaggle.com/datasets/thefish81/beige-books/data) to download the data.

# Model Reproduction:
## Bert Model:
To train the BERT regressor run: `python -m Scripts.bert_train -h`
## FOMC Models:
- To reproduce the FOMC models, run the following command from the base of the repo:
`python -m Scripts.fomc_train '2008-01-01' '2024-07-01' --model philschmid/bge-base-financial-matryoshka`
- To evaluate and create the plots:
`python -m Scripts.fomc_test`

# Viewing the report:
1. [download](https://download-directory.github.io/?url=https%3A%2F%2Fgithub.com%2FTheFish18%2FMADSCapstone%2Ftree%2Fmain%2Fmain) main/
2. open main.html
