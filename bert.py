'''
Last modified on November 27th, 2020
Trains (or imports from previous run) a BERT model to assign sentiment scores to financial news headlines
@Zachary Lee
'''
import os
from pathlib import Path

import nltk
import pandas as pd
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from textblob import TextBlob, Sentence

from finbert.finBERT.finbert.finbert import predict
from train_bert import train_bert


def calculate_bert_sentiment(df_headlines, b_from_file=False):

    if(os.path.exists('df_bert_headlines.csv') and b_from_file):
        print("Reading bert sentiment from file to save time!")
        return pd.read_csv('df_bert_headlines.csv')
    elif not os.path.exists('df_bert_headlines.csv') and b_from_file:
        raise FileNotFoundError("ERROR: file reading enabled, but \"df_bert_headlines.csv\" not found!")

    project_dir = str(Path.cwd()) + "/finbert/finBERT"

    if not os.path.exists(project_dir + "/models/classifier_model/finbert-sentiment"):
        print("No model found, commencing training... please be patient!")
        train_bert()

    print("trained BERT sentiment model found... calculating headline sentiment")
    nltk.download('punkt')
    df_bert = get_sentiment(df_headlines)
    df_headlines = df_headlines.merge(df_bert, on='title', how='inner')

    df_logit = pd.DataFrame(df_headlines[['title','logit']])
    df_logit[['pos_prob', 'neu_prob', 'neg_prob']] = pd.DataFrame(df_logit['logit'].tolist(), index=df_logit.index)
    df_final_logit = pd.DataFrame(df_logit['logit'].tolist(), columns=['pos_prob', 'neu_prob', 'neg_prob'])
    df_final_logit['title'] = df_logit['title']

    del df_headlines['logit']

    df_headlines = df_headlines.merge(df_final_logit, on='title', how='inner')

    df_headlines.to_csv('df_bert_headlines.csv')
    return df_headlines

def get_sentiment(df_headlines):
    ls_titles = df_headlines['title'].tolist()
    project_dir = str(Path.cwd()) + "/finbert/finBERT"
    cl_path = project_dir + "/models/classifier_model/finbert-sentiment"
    model = BertForSequenceClassification.from_pretrained(cl_path, cache_dir=None, num_labels=3)


    ldf_bert_title_sentiments = [predict(title, model) for title in ls_titles]
    df_bert_title_sentiments = pd.concat(ldf_bert_title_sentiments)
    df_bert_title_sentiments.reset_index(inplace=True, drop=True)
    # use TextBlob to seperate string into sentences, then evaluate their sentiment using finbert
    blob = TextBlob(ls_titles[0])
    for title in ls_titles[1:]:
        blob.sentences.append(Sentence(title))
    ss_titles = pd.Series([sentence.raw for sentence in blob.sentences])
    sf_title_sentiment = pd.Series([sentence.sentiment.polarity for sentence in blob.sentences])
    df_textblob_title_sentiments = pd.DataFrame()
    df_textblob_title_sentiments['title'] = ss_titles
    df_textblob_title_sentiments['textblob_sentiment_prediction'] = sf_title_sentiment
    i_temp_len = len(df_bert_title_sentiments)

    df_bert_title_sentiments['title'] = df_bert_title_sentiments['sentence']
    del df_bert_title_sentiments['sentence']

    df_bert_title_sentiments = df_bert_title_sentiments.merge(df_textblob_title_sentiments, on='title', how='inner')
    print(str((len(ldf_bert_title_sentiments)/i_temp_len)*100) + "% of berts/textblob sentiments merged!")
    print("Tuned BERT model complete!! " + str(len(df_bert_title_sentiments)) + " financial headlines have been processed successfully!")
    print(f'Average headline sentiment is %.2f.' % (df_bert_title_sentiments.sentiment_score.mean()))

    return df_bert_title_sentiments
