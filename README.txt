Language Models for Financial Sentiment and Stock Prediction Using News Flows
CSC 248 Final Project Documentation

Data Files:

The market relative returns data file is too large to upload, so I uploaded a small sample called 5_day_return_test.csv

headlines.csv contains the data from web scraping, as described in the paper

Other .csv files are intermediatly saved as each model trains to provide memoziation optionality and speed up debugging

Main:

main.py contains the high level overview of the entire project, except the roc auc plots, which were generated independently using roc_auc.py

train_bert.py is a special module that can be called to fine tune a custom BERT model on your own dataset, as I did in this project. This was inspired by 
some code taken from the FinBERT library repository located here https://github.com/ProsusAI/finBERT, but several changes were made to make the model
compatible and effective with the Benzinga dataset.

bert.py imports the trained model from train_bert.py and runs it on an input dataset, generating sentiment scores. bert.py also contains the naive bayes 
implementation, which I admit is confusing.

nlp_utils contains utility functions that are mainly used in the model files (bert.py, train_bert.py, and vader.py)

To run the code, the following libraries must be installed:
channels:
  - pytorch
  - anaconda
  - conda-forge
  - defaults
dependencies:
  - jupyter=1.0.0
  - pandas=0.23.4
  - python=3.7.3
  - numpy==1.16.3
  - nltk
  - tqdm
  - ipykernel=5.1.3
  - textblob
  - pip:
    - joblib==0.13.2
    - pytorch-pretrained-bert==0.6.2
    - scikit-learn==0.21.2
    - spacy==2.1.4
    - torch==1.1.0
	- vaderSentiment
	
At that point, try running main to train your first model on a very cool dataset! 
