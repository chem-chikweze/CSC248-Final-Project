import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from pandas_ml import ConfusionMatrix
from sklearn import preprocessing
from sklearn.metrics import roc_curve

# before we create the binary classifier, we need to remove ensemble rows that do not contain sentiment or performance information
df_asset_sample = df_asset_sample.dropna(
    subset=['vader_compound_sentiment', 'naive_compound_sentiment', 'finbert_compound_sentiment', 'value'])
df_asset_sample = df_asset_sample[df_asset_sample['vader_compound_sentiment'] != 0]
df_asset_sample = df_asset_sample[df_asset_sample['naive_compound_sentiment'] != 0]
df_asset_sample = df_asset_sample[df_asset_sample['finbert_compound_sentiment'] != 0]
df_asset_sample = df_asset_sample.reset_index()
del df_asset_sample['index']
df_asset_sample['ensemble_compound_sentiment'] = df_asset_sample[
    ['vader_compound_sentiment', 'naive_compound_sentiment', 'finbert_compound_sentiment']].mean(axis=1)
df_asset_sample.to_csv("sentiment_return_value_data.csv")

# predicted direction depends on the aggregate sentiment
df_asset_sample['y_Predicted_vader'] = df_asset_sample['vader_compound_sentiment'] > 0
df_asset_sample['y_Predicted_finbert'] = df_asset_sample['finbert_compound_sentiment'] > 0
df_asset_sample['y_Predicted_naive'] = df_asset_sample['naive_compound_sentiment'] > 0

df_asset_sample['y_Predicted_ensemble'] = ((df_asset_sample['vader_compound_sentiment'] > 0) \
                                           & (df_asset_sample['finbert_compound_sentiment'] > 0) & (
                                                   df_asset_sample['naive_compound_sentiment'] > 0))

df_asset_sample['y_Predicted_mean'] = df_asset_sample['ensemble_compound_sentiment'] > 0

# create binary target labels, with True representing an outperforming asset and False represent an underperforming asset
df_asset_sample['y_Actual'] = df_asset_sample['value'] > 0

# convert booleans to integers to create a standard binary classifier and add 1 so probabilities can be used
df_asset_sample['y_Predicted_vader'] = df_asset_sample['y_Predicted_vader'].astype(int)
df_asset_sample['y_Predicted_finbert'] = df_asset_sample['y_Predicted_finbert'].astype(int)
df_asset_sample['y_Predicted_naive'] = df_asset_sample['y_Predicted_naive'].astype(int)
df_asset_sample['y_Predicted_ensemble'] = df_asset_sample['y_Predicted_ensemble'].astype(int)
df_asset_sample['y_Predicted_mean'] = df_asset_sample['y_Predicted_mean'].astype(int)
df_asset_sample['y_Actual'] = df_asset_sample['y_Actual'].astype(int)

# The farther the sentiment is from 0, the higher the confidence in the prediction
x = df_asset_sample[['vader_compound_sentiment']].values.astype(float)
min_max_scaler_x = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler_x.fit_transform(x)
df_asset_sample['y_Confidence_vader'] = x_scaled

df_asset_sample['y_Confidence_vader_bullish'] = np.nan_to_num((df_asset_sample[['vader_compound_sentiment']].mask(
    df_asset_sample['vader_compound_sentiment'].lt(0)).values.astype(float)) / 100)
df_asset_sample['y_Confidence_vader_bearish'] = np.nan_to_num((df_asset_sample[['vader_compound_sentiment']].mask(
    df_asset_sample['vader_compound_sentiment'].gt(0)).values.astype(float)) / -100)

# The farther the sentiment is from 0, the higher the confidence in the prediction
y = df_asset_sample[['finbert_compound_sentiment']].values.astype(float)
min_max_scaler_y = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler_y.fit_transform(y)
df_asset_sample['y_Confidence_finbert'] = y_scaled

df_asset_sample['y_Confidence_finbert_bullish'] = np.nan_to_num((df_asset_sample[['finbert_compound_sentiment']].mask(
    df_asset_sample['finbert_compound_sentiment'].lt(0)).values.astype(float)) / 100)
df_asset_sample['y_Confidence_finbert_bearish'] = np.nan_to_num((df_asset_sample[['finbert_compound_sentiment']].mask(
    df_asset_sample['finbert_compound_sentiment'].gt(0)).values.astype(float)) / -100)

# The farther the sentiment is from 0, the higher the confidence in the prediction
z = df_asset_sample[['naive_compound_sentiment']].values.astype(float)
min_max_scaler_z = preprocessing.MinMaxScaler()
z_scaled = min_max_scaler_z.fit_transform(z)
df_asset_sample['y_Confidence_naive'] = z_scaled

df_asset_sample['y_Confidence_naive_bullish'] = np.nan_to_num((df_asset_sample[['naive_compound_sentiment']].mask(
    df_asset_sample['naive_compound_sentiment'].lt(0)).values.astype(float)) / 100)
df_asset_sample['y_Confidence_naive_bearish'] = np.nan_to_num((df_asset_sample[['naive_compound_sentiment']].mask(
    df_asset_sample['naive_compound_sentiment'].gt(0)).values.astype(float)) / -100)

# The farther the sentiment is from 0, the higher the confidence in the prediction
b = df_asset_sample[['ensemble_compound_sentiment']].values.astype(float)
min_max_scaler_b = preprocessing.MinMaxScaler()
b_scaled = min_max_scaler_b.fit_transform(b)
df_asset_sample['y_Confidence_ensemble_mean'] = b_scaled

df_asset_sample['y_Confidence_ensemble_mean_bullish'] = np.nan_to_num(
    (df_asset_sample[['ensemble_compound_sentiment']].mask(
        df_asset_sample['ensemble_compound_sentiment'].lt(0)).values.astype(float)) / 100)
df_asset_sample['y_Confidence_ensemble_mean_bearish'] = np.nan_to_num(
    (df_asset_sample[['ensemble_compound_sentiment']].mask(
        df_asset_sample['ensemble_compound_sentiment'].gt(0)).values.astype(float)) / -100)

# TODO filter df_asset_sample by model

generate_confusion_matrix_stats('vader', df_asset_sample['y_Actual'], df_asset_sample['y_Predicted_vader'], s_model)
generate_confusion_matrix_stats('finbert', df_asset_sample['y_Actual'], df_asset_sample['y_Predicted_finbert'],
                                s_model)
generate_confusion_matrix_stats('naive', df_asset_sample['y_Actual'], df_asset_sample['y_Predicted_naive'], s_model)
generate_confusion_matrix_stats('ensemble', df_asset_sample['y_Actual'], df_asset_sample['y_Predicted_ensemble'],
                                s_model)
generate_confusion_matrix_stats('mean', df_asset_sample['y_Actual'], df_asset_sample['y_Predicted_mean'], s_model)

generate_roc_prob_curves('vader', df_asset_sample['y_Confidence_vader_bullish'],
                         df_asset_sample['y_Confidence_vader_bearish'], s_model)
generate_roc_prob_curves('naive', df_asset_sample['y_Confidence_naive_bullish'],
                         df_asset_sample['y_Confidence_naive_bearish'], s_model)
generate_roc_prob_curves('finbert', df_asset_sample['y_Confidence_finbert_bullish'],
                         df_asset_sample['y_Confidence_finbert_bearish'], s_model)
generate_roc_prob_curves('ensemble_mean', df_asset_sample['y_Confidence_ensemble_mean_bullish'],
                         df_asset_sample['y_Confidence_ensemble_mean_bearish'], s_model)

# calculate Receiver operating characteristic (ROC) and Area Under Curve (AUC)
ns_probs = [0 for _ in range(len(df_asset_sample))]
ns_fpr, ns_tpr, _ = roc_curve(df_asset_sample['y_Actual'], ns_probs)
fpr_vader, tpr_vader, thresholds = roc_curve(df_asset_sample['y_Actual'], df_asset_sample['y_Confidence_vader'])
fpr_finbert, tpr_finbert, thresholds = roc_curve(df_asset_sample['y_Actual'], df_asset_sample['y_Confidence_finbert'])
fpr_naive, tpr_naive, thresholds = roc_curve(df_asset_sample['y_Actual'], df_asset_sample['y_Confidence_naive'])
fpr_ensemble, tpr_ensemble, thresholds = roc_curve(df_asset_sample['y_Actual'],
                                                   df_asset_sample['y_Confidence_ensemble_mean'])

# auc = roc_auc_score(df['y_Actual'], df_asset_sample['y_Confidence'])
# print("Area Under Curve: ")
# print(auc)

# plot the roc curve for the model
plt.figure()
plt.plot(ns_fpr, ns_tpr, linestyle='--', label="Random")
plt.plot(fpr_vader, tpr_vader, label="vader")
plt.plot(fpr_finbert, tpr_finbert, label="finbert")
plt.plot(fpr_naive, tpr_naive, label="naive")
plt.plot(fpr_ensemble, tpr_ensemble, label="ensemble")
# TODO add in the results using other ML Models to compare

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('roc_' + str(i_prediction_horizon) + '_' + s_model + '.pdf')
plt.savefig('roc_' + str(i_prediction_horizon) + '_' + s_model + '.png')


def generate_confusion_matrix_stats(s_name, si_actual, si_prediction, s_model):
    confusion_matrix = pd.crosstab(si_actual, si_prediction, rownames=['Actual'],
                                   colnames=['Predicted'])
    plt.figure()
    sn.heatmap(confusion_matrix, annot=True, fmt='g')
    plt.savefig('confusion_matrix_ ' + s_name + '_' + str(i_prediction_horizon) + '_' + s_model + '.pdf')
    plt.savefig('confusion_matrix_ ' + s_name + '_' + str(i_prediction_horizon) + '_' + s_model + '.png')

    # generate and save additional ML stats
    Confusion_Matrix = ConfusionMatrix(si_actual, si_prediction)
    original = sys.stdout
    sys.stdout = open("ml_stats_" + s_name + '_' + str(i_prediction_horizon) + '_' + s_model + ".txt", "w")

    print(Confusion_Matrix.print_stats())
    sys.stdout = original

def generate_roc_prob_curves(s_name, si_bullish_probs, si_bearish_probs, s_model):
    si_bullish_probs = si_bullish_probs.values
    si_bearish_probs = si_bearish_probs.values
    x = np.vstack((si_bullish_probs, si_bearish_probs)).T
    pdf(x, s_name, s_model)

def pdf(x, s_name, s_model):
    plt.figure()

    labels = ['Bullish Probability', 'Bearish Probability']
    plt.figure(dpi=150)

    plt.hist(x, bins=20,label=labels)

    plt.title('Classification Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('# of Instances')
    plt.xlim([0.5, 1.0])
    plt.autoscale()

    # avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig('pdf_'+ s_name + "_" + str(i_prediction_horizon) + '_' + s_model + '.pdf')
    plt.savefig('pdf_' + s_name + "_" + str(i_prediction_horizon) + '_' + s_model + '.png')

