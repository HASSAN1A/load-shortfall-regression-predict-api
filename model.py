"""
    Helper functions for the pretrained model to be used within our API.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------
    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.
"""

# Helper Dependencies
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import warnings
import statsmodels.formula.api as sm
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.graphics.correlation import plot_corr
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import missingno
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pandas as pd
import pickle
import json
# Import Dependencies
%matplotlib inline

# Start Python Imports

# Data Manipulation

# Visualization

%matplotlib inline

# Preprocessing

# Machine learning
# from catboost import CatBoostClassifier, Pool, cv
# Buid the Model
# OLS summary
# Let's be rebels and ignore warnings for now
warnings.filterwarnings('ignore')


def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.
    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.
    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------
    train = pd.read_csv('data/df_train.csv')     test = pd.read_csv('data/df_test.csv')
    # example of what a submission should look like
    ls_submission = pd.read_csv('data/sample_submission_load_shortfall.csv')

# Insurance dataset
    copy_train = train.copy()
    copy_test = test.copy()
    missing_train_values = train.Valencia_pressure.isna().sum()
    print(
        f'Missing train Values: {missing_train_values}  |   Percentage: {round(( missing_train_values/ train.Valencia_pressure.shape[0]) *100, 2)}%')
    missing_values_test = test.Valencia_pressure.isna().sum()
    print(
        f'Test data Missing Values: {missing_values_test}  |   Percentage: {round(( missing_values_test / train.Valencia_pressure.shape[0]) *100, 2)}%')
    mode = pd.concat([train.Valencia_pressure, test.Valencia_pressure]).mode()
    # Impute missing values in Valencia_pressure with mean
    train.Valencia_pressure.fillna(mode[0], inplace=True)
    test.Valencia_pressure.fillna(mode[0], inplace=True)
    missing_values_train = train.Valencia_pressure.isna().sum()
    print(
        f'Train data Missing Values: {missing_values_train}  |   Percentage: {round(( missing_values_train/ train.Valencia_pressure.shape[0]) *100, 2)}%')
    print(
        f'Sum of unique object: {train.Valencia_wind_deg.value_counts().count()}')
    train.Valencia_wind_deg.unique()
    print(
        f'Sum of unique object: {train.Seville_pressure.value_counts().count()}')
    train.Seville_pressure.unique()
    from sklearn.preprocessing import OrdinalEncoder
    # Impute Categorical features using OrdinalEncoder()
    enc = OrdinalEncoder()
    train.Valencia_wind_deg = enc.fit_transform(train[['Valencia_wind_deg']])
    train.Seville_pressure = enc.fit_transform(train[['Seville_pressure']])
    test.Valencia_wind_deg = enc.fit_transform(test[['Valencia_wind_deg']])
    test.Seville_pressure = enc.fit_transform(test[['Seville_pressure']])

    import datetime as dt

    train['time'] = pd.to_datetime(train['time'])
    test['time'] = pd.to_datetime(test['time'])
    test_copy = test.copy()

    # day
    train['Day'] = train['time'].dt.day
    test['Day'] = test['time'].dt.day

    # month
    train['Month'] = train['time'].dt.month
    test['Month'] = test['time'].dt.month

    # year
    train['Year'] = train['time'].dt.year
    test['Year'] = test['time'].dt.year

    # hour
    train['Start_hour'] = train['time'].dt.hour
    test['Start_hour'] = test['time'].dt.hour

    # Drop Feature
    train.drop(['time'], axis=1, inplace=True)
    test.drop(['time'], axis=1, inplace=True)
    columns = train.drop(['load_shortfall_3h'], axis=1).columns
    # Scale the dataset
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        train.drop(['load_shortfall_3h'], axis=1).values)
    scaled_features_test = scaler.fit_transform(test.values)
    train_scaled = pd.DataFrame(
        scaled_features, index=train.index, columns=columns)
    test_scaled = pd.DataFrame(
        scaled_features_test, index=test.index, columns=columns)
    # Add load_short_fall_3h as last_columns on training data
    train_scaled['load_shortfall_3h'] = copy_train.load_shortfall_3h.values
    # Perform a test_train_split
    X = train_scaled.drop(['load_shortfall_3h'], axis=1)
    y = train_scaled.load_shortfall_3h
    # Train Test Split
    X = train.drop(['load_shortfall_3h'], axis=1)
    y = train.load_shortfall_3h

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    select_important = SelectFromModel(RandomForestRegressor(
        n_estimators=100, random_state=0, n_jobs=1))
    select_important.fit(X_train, y_train)
    select_important.get_support()
    features = X_train.columns[(select_important.get_support())]
    len(features)
    np.mean(select_important.estimator_.feature_importances_)
    pd.Series(select_important.estimator_.feature_importances_.ravel()).hist()
    select_important.estimator_.feature_importances_
    X_train_sel = select_important.transform(X_train)
    X_test_sel = select_important.transform(X_test)

    def _randomForest_train(X_train, X_test, y_train, y_test):
        forest = RandomForestRegressor(
            n_estimators=100, random_state=0, n_jobs=1)
        forest.fit(X_train, y_train)
        pred = forest.predict(X_test)
        print(
            f'Train RMSE: { np.sqrt(mean_squared_error(y_train[:2892], pred))}')
        print(f'R Score Train: {r2_score(y_train[:2892] , pred)}')
        print(f'Test RMSE: { np.sqrt(mean_squared_error(y_test, pred))}')
        print(f'R Score Test: {r2_score(y_test , pred)}')
    _randomForest_train(X_train, X_test, y_train, y_test)
    r_selection = RFE(RandomForestRegressor(n_estimators=100,
                                            random_state=0, n_jobs=1), n_features_to_select=14)
    r_selection.fit(X_train, y_train)

    # ------features1 = X_train.columns[(r_selection.get_support())]
    len(features)
    grad_selection = RFE(GradientBoostingRegressor(
        n_estimators=100, random_state=0), n_features_to_select=10)
    grad_selection.fit(X_train, y_train)
    X_train_g = grad_selection.transform(X_train)
    X_test_g = grad_selection.transform(X_test)
    _randomForest_train(X_train_g, X_test_g, y_train, y_test)
    _randomForest_train(X_train, X_test, y_train, y_test)
    print('~'*40 + 'GradientBoostRegressor' + '~'*40)
    for index in range(1, 51):
        grad_selection = RFE(GradientBoostingRegressor(
            n_estimators=100, random_state=0), n_features_to_select=index)
        grad_selection.fit(X_train, y_train)
        X_train_g = grad_selection.transform(X_train)
        X_test_g = grad_selection.transform(X_test)
        print(f'Selected Features: {index}')
        _randomForest_train(X_train_g, X_test_g, y_train, y_test)
        print()
    print('~'*40 + 'RandomForestRegressor' + '~'*40)
    for index in range(1, 51):
        grad_selection = RFE(RandomForestRegressor(
            n_estimators=100, random_state=0), n_features_to_select=index)
        grad_selection.fit(X_train, y_train)
        X_train_r = grad_selection.transform(X_train)
        X_test_r = grad_selection.transform(X_test)
        print(f'Selected Features: {index}')
        _randomForest_train(X_train_r, X_test_r, y_train, y_test)
        print()
    grad_sel = RFE(GradientBoostingRegressor(
        n_estimators=100, random_state=0), n_features_to_select=4)
    grad_sel.fit(X_train, y_train)
    X_train_g = grad_sel.transform(X_train)
    X_test_g = grad_sel.transform(X_test)
    print(f'Selected Features: {4}')
    _randomForest_train(X_train_g, X_test_g, y_train, y_test)
    best_feat1 = X_train.columns[(grad_sel.get_support())]
    len(best_feat1)
    forest_selection = RFE(RandomForestRegressor(
        n_estimators=100, random_state=0, n_jobs=1), n_features_to_select=10)
    forest_selection.fit(X_train, y_train)
    X_train_g = forest_selection.transform(X_train)
    X_test_g = forest_selection.transform(X_test)
    print(f'Selected Features: {10}')
    _randomForest_train(X_train_g, X_test_g, y_train, y_test)
    feature_vector_df = ['Day', 'Month', 'Year', 'Start_hour']

    predict_vector = feature_vector_df[[
        ['Day', 'Month', 'Year', 'Start_hour']]]
    return predict_vector


def load_model(path_to_model: str):
    """Adapter function to load our pretrained model into memory.
    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.
    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.
    """
    return pickle.load(open(path_to_model, 'rb'))


def make_prediction(data, model):
    """Prepare request data for model prediciton.
    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.
    Returns
    -------
    list
        A 1-D python list containing the model prediction.
    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction.tolist()
