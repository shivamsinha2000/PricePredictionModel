# Group 1: Thomas Kantaros, Shivam Sinha, Gregor
# 12/10/24

import pandas as pd
from sklearn.linear_model import LinearRegression

def drop_cols(df, columns):
    """
    Reads in a dataframe and columns and drops the columns
    :param df: the pandas dataframe
    :param columns: the list of columns to drop
    :return: a clean dataframe with columns dropped
    """
    new_df = df.drop(columns, axis=1)
    return new_df

def set_categories(df, cat_columns):
    """
    Reads in a dataframe and columns and sets them as categories 
    :param df: the pandas dataframe
    :param cat_columns: the list of columns
    :return: a clean dataframe with categories set
    """

    for col in cat_columns:
        df[col] = df[col].astype('category')
    return df

def linear_reg(X, y):
    """
    Reads in two columns and runs linear regression
    :param X: the predictor
    :param y: the target
    :return: a fitted linear regression model
    """

    reg = LinearRegression()
    reg.fit(X, y)
    return reg



