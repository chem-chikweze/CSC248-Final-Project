'''
Last modified on November 28th, 2020
Evaluates the efficacy of 3 methods of generating sentiment for financial news headlines,
with the purpose of predicting market relative returns of the specific stock mentioned.
1) pre-trained generic VADER model
2) BERT model fine tuned to the financial domain
3) Textblob Naive Bayes polarity score using most significant finance keywords
@Zachary Lee
'''

from bert import calculate_bert_sentiment
from nlp_util import get_mapped_headlines, merge_returns
from vader import calculate_vader_sentiment

if __name__ == '__main__':

    # create dataframe of financial news headlines mapped to a single stock ticker that is mentioned within them
    df_headlines = get_mapped_headlines(0.0533, b_from_file=True)
    df_headlines = merge_returns(df_headlines, b_from_file=True)

    # calculate the sentiment types
    df_headlines = calculate_vader_sentiment(df_headlines, b_from_file=False)
    df_headlines = calculate_bert_sentiment(df_headlines, b_from_file=False)

    # create ROC binary classifier and plot the AUC of each langauge model
    
