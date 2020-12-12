'''
Last modified on November 27th, 2020
Contains functions that enable the parallelized calculation of VADER sentiment for financial news headlines
@Zachary Lee
'''

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from functools import partial
from nlp_util import parallelize_on_rows
import os
import pandas as pd


# calculates positive, negative, and neutral vader sentiment on a dataframe of financial news headlines
def calculate_vader_sentiment(df_headlines, b_from_file=False):
    if(os.path.exists('df_vader_headlines.csv') and b_from_file):
        print("Reading vader sentiment from file to save time!")
        return pd.read_csv('df_vader_headlines.csv')
    elif not os.path.exists('df_vader_headlines.csv') and b_from_file:
        raise FileNotFoundError("ERROR: file reading enabled, but \"df_vader_headlines.csv\" not found!")

    # get compound sentiment scores for each title using VADER
    print("Calculating VADER sentiment scores (may take several minutes)")
    sid = SentimentIntensityAnalyzer()

    get_positive_sentiment = partial(get_sentiment, k='positive', sid=sid)
    get_negative_sentiment = partial(get_sentiment, k='negative', sid=sid)
    get_compound_sentiment = partial(get_sentiment, k='compound_sentiment', sid=sid)

    # get generic VADER Sentiment
    df_headlines['vader_title_positive'] = parallelize_on_rows(df_headlines['title'], get_positive_sentiment)
    df_headlines['vader_title_negative'] = parallelize_on_rows(df_headlines['title'], get_negative_sentiment)
    df_headlines['vader_title_compound_sentiment'] = parallelize_on_rows(df_headlines['title'], get_compound_sentiment)

    # aggregate sentiment by ticker and day in case a ticker has multiple articles released per day
    # Note: this df still has 0 values, which are 40% of the dataset being factored into the mean()
    df_headlines = df_headlines.groupby(['updated_day', 'ticker', 'title']).mean()
    df_headlines.reset_index(inplace=True)
    df_headlines.to_csv('df_vader_headlines.csv')
    print("VADER sentiment calculation complete, " + str(len(df_headlines)) + " headlines have been processed!\n")

    return df_headlines


def get_sentiment(row, sid, **kwargs):
    sentiment_score = sid.polarity_scores(row)
    positive_meter = round((sentiment_score['pos'] * 100), 2)
    negative_meter = round((sentiment_score['neg'] * 100), 2)
    compound_meter = round((sentiment_score['compound'] * 100), 2)
    if kwargs['k'] == 'compound_sentiment':
        return compound_meter
    elif kwargs['k'] == 'positive':
        return positive_meter
    elif kwargs['k'] == 'negative':
        return negative_meter
    else:
        raise AttributeError
