'''
Last modified on October 22nd, 2020
Contains some utility functions used to import and preprocess project data
@Zachary Lee
'''

import pandas as pd
from functools import partial
from multiprocessing import Pool
import numpy as np
import os


def get_headlines():
    df_headlines = pd.read_csv('headlines.csv')
    df_headlines.drop(df_headlines.columns[[0]], axis=1, inplace=True)
    return df_headlines


def get_headline_assets():
    df_headline_assets = pd.read_csv('headline_asset.csv')
    df_headline_assets.drop(df_headline_assets.columns[[0]], axis=1, inplace=True)
    return df_headline_assets


def get_headline_categories():
    df_headline_categories = pd.read_csv('headline_category.csv')
    df_headline_categories.drop(df_headline_categories.columns[[0]], axis=1, inplace=True)
    return df_headline_categories


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


# merges the three headline datasets into one dataframe for easier analysis
def get_mapped_headlines(f_subset=1.0, b_from_file=False):

    if(os.path.exists('df_single_headlines.csv') and b_from_file):
        print("Reading single headlines from file to save time!")
        return pd.read_csv('df_single_headlines.csv')
    elif not os.path.exists('df_single_headlines.csv') and b_from_file:
        raise FileNotFoundError("ERROR: file reading enabled, but \"df_single_headlines.csv\" not found!")

    print("Initializing headline/return data (may take a minute)\n")

    df_headlines = get_headlines()
    df_headline_assets = get_headline_assets()
    df_headline_categories = get_headline_categories()

    df_headline_categories = df_headline_categories.drop_duplicates('headline_id')
    del df_headline_categories['id']
    del df_headline_assets['id']
    df_headline_categories['id'] = df_headline_categories['headline_id']
    df_headline_assets['id'] = df_headline_assets['headline_id']
    del df_headline_categories['headline_id']
    del df_headline_assets['headline_id']

    # only looking at articles mentioning single assets for now

    df_headlines = df_headlines.merge(df_headline_assets, on=['id'], how='right')
    # we don't want to add or subtract any rows with this merge
    df_headlines = df_headlines.merge(df_headline_categories, on=['id'], how='inner')

    df_headlines = df_headlines[['id', 'updated_date', 'title', 'assetid', 'ticker', 'name']]

    # drop headlines that were not correctly mapped to a ticker
    df_single_ticker_headlines = df_headlines[df_headlines['assetid'].notnull()].drop("assetid", axis=1)
    if f_subset < 1.0:
        df_single_ticker_headlines = df_single_ticker_headlines.sample(frac=f_subset, random_state=1)
    df_single_ticker_headlines = df_single_ticker_headlines.dropna(subset=['title'])
    # print("mapping rate: " + str(len(df_single_ticker_headlines)/len(df_headlines)))
    df_single_ticker_headlines = df_single_ticker_headlines.drop_duplicates(subset=['id'], keep=False)

    df_single_ticker_headlines.to_csv('df_single_headlines.csv')
    return df_single_ticker_headlines


def parallelize_on_rows(data, func, num_of_processes=26):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


def parallelize(data, func, num_of_processes=26):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


# creates a dataframe of ticker popularity based on the number of headlines
# written about a ticker on a day relative to a 2 month average number of daily headlines for that ticker
def ticker_popularity(df_asset_sample):
    df_asset_sample = df_asset_sample[['updated_date', 'ticker']]
    df_asset_sample['updated_day'] = df_asset_sample['updated_date'].dt.date
    del df_asset_sample['updated_date']

    # calculate number of mentions per ticker per day
    df_asset_sample = pd.DataFrame(df_asset_sample.groupby(['updated_day', 'ticker'])['ticker'].count())

    # rolling two month average number of daily articles per ticker
    df_asset_sample = df_asset_sample.unstack()
    df_asset_sample = df_asset_sample.fillna(0)

    df_rolling_average = df_asset_sample.copy()
    df_rolling_average = df_rolling_average.rolling(window=42, center=False).mean()

    df_cum_sum = df_asset_sample.cumsum()

    df_ticker_popularity = df_asset_sample / df_rolling_average
    df_ticker_popularity.to_csv('ticker_popularity.csv')


# returns a dataframe of 5-day market relative returns by day by ticker for tickers mentioned within news headlines
def get_market_relative_returns():
    df_returns = pd.read_csv('5_day_rel_return.csv')
    # shift by -5 because we want our labels to be in the future. So the return value on a given day
    # represents the cumulative relative return of the stock to the S&P500 over the NEXT 5 days
    # this represents our y label, which makes sense as we are trying to predict 5 days into the future
    mask = ~(df_returns.columns.isin(['day']))
    cols_to_shift = df_returns.columns[mask]
    df_returns[cols_to_shift] = df_returns.loc[:, mask].shift(-5)

    return df_returns


# converts objects into datetimes and adds days column
def parse_dates(df_headlines):
    df_headlines['updated_date'] = pd.to_datetime(df_headlines['updated_date'])
    df_headlines['updated_day'] = df_headlines['updated_date'].dt.date
    return df_headlines


# prepare market relative returns dataframe for merging with headlines dataframe
def melt_returns(df_returns):
    df_returns = df_returns.melt(id_vars=['day'])
    df_returns['return_id'] = df_returns['day'].astype(str) + df_returns['variable'].astype(str)
    df_returns = df_returns[['return_id', 'value']]
    return df_returns


# merges the above returns into the existing headlines dataframe, as needed
def merge_returns(df_headlines, b_from_file=False):

    if(os.path.exists('df_merged_headlines.csv') and b_from_file):
        print("Reading merged headlines from file to save time!")
        return pd.read_csv('df_merged_headlines.csv')
    elif not os.path.exists('df_merged_headlines.csv') and b_from_file:
        raise FileNotFoundError ("ERROR: file reading enabled, but \"df_merged_headlines.csv\" not found!")

    df_returns = get_market_relative_returns()

    df_headlines = df_headlines.reset_index()
    del df_headlines['index']

    df_headlines = parse_dates(df_headlines)

    df_headlines['return_id'] = df_headlines['updated_day'].astype(str) + df_headlines['ticker'].astype(str)

    df_headlines = df_headlines.merge(melt_returns(df_returns), on='return_id', how='inner')
    del df_headlines['return_id']
    # discard articles that cannot be linked to return values as they will be useless
    df_headlines = df_headlines.dropna(subset=['value'])
    df_headlines.to_csv('df_merged_headlines.csv')

    return df_headlines
