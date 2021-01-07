#common code:
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import re
#week 3
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
#week 4
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
#week 5
import requests
from bs4 import BeautifulSoup
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

def pipemaker(scaler, classifier):
    if scaler == 'cvec':
        item1 = ('cvec', CountVectorizer())
    elif scaler == 'tvec':
        item1 = ('tvec', TfidfVectorizer())
    else:
        return 'Error. Please enter "cvec" or "tvec"'
    if classifier == 'nb':
        item2 = ('nb', MultinomialNB())
    elif classifier == 'rf':   #added for model 10
        item2 = ('rf', RandomForestClassifier())
    elif classifier == 'logreg':#updated for model 12
        item2 = ('logreg', LogisticRegression())
    else:
        return 'Error.'
    return Pipeline([item1, item2])


def set_params(scaler, max_feat_list, min_df_list, max_df_list, sw_list, ngram_list):
    if scaler == 'cvec' or scaler == 'tvec':
        pipe_params = {f'{scaler}__max_features': max_feat_list,
                  f'{scaler}__min_df': min_df_list,
                  f'{scaler}__max_df': max_df_list,
                  f'{scaler}__stop_words': sw_list,
                  f'{scaler}__ngram_range': ngram_list}
    else:
        return "sorry, please use cvec or tvec"
    return pipe_params


def score_model(model, X_train, y_train, X_test, y_test):
    print(f'The best parameters are: {model.best_params_}')
    print(f'The best training score was: {model.best_score_}')
    print(f'The test score is: {model.score(X_test, y_test)}')
    preds = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print('_'*20)
    print('Confusion Matrix for Test Set:')
    plot_confusion_matrix(model, X_test, y_test, cmap='Blues', values_format='d');
    plt.show()
    print(f'The Accurracy score is {metrics.accuracy_score(y_test, preds)}')
    print(f'The Sensitivity score is {metrics.recall_score(y_test, preds)}')
    print(f'The Precision score is {metrics.precision_score(y_test, preds)}')
    print('_'*20)
    print('Receiver Operating Characteristic (ROC) curve:')
    metrics.plot_roc_curve(model, X_test, y_test);
    plt.show()
    return 'Model scored!'