# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.mixture import GMM


dc = {'lead':0, 'short.lag':1, 'long.lag':2}
da = {'labial':0,'coronal':1, 'dorsal':2}

def avg_data_reader(fname='data/ChodroffGoldenWilson2019_vot_avg.csv'):
    """reads in data from ChodroffGoldenWilson2019_vot_avg.csv,
    expected format: family,language,dialect,vot.category,poa2,vot.mu

    discards languages without all 3 poa per vot.category"""

    full = pd.read_csv('data/ChodroffGoldenWilson2019_vot_avg.csv', delimiter = ',', na_filter=False)            
    full['vot.category'] = full['vot.category'].map(dc)
    full['poa2'] = full['poa2'].map(da)
    full['lang'] = full[['family', 'language', 'dialect']].apply(lambda x: '_'.join(x), axis=1)
    full = full.drop(['family', 'language', 'dialect'], axis=1)
    grouped = full.groupby(['lang','vot.category'])
    grouped = grouped.filter(lambda x: len(x) == 3)
    grouped = grouped.loc[grouped['vot.category'] == dc['short.lag']]

    data  = grouped.values[:,0:3]
    langs = grouped.values[:,3]

    return data, langs

#def raw_data_reader(fname='data/ChodroffGoldenWilson2019_vot.csv'):
#    """reads in data from ChodroffGoldenWilson2019_vot.csv,
#    expected format: family,language,dialect,source,primarySource,poa1,poa2,vot.category,vot,voiceContrast,notes
#    """


def data_split(X, y, train_split=0.8, dev_split=0.1, test_split=0.1):
    """splits data into train, dev, test sets"""
    assert train_split+dev_split+test_split == 1, "data splits do not add up to 1"

     X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_split, random_state=1)
     X_train, X_val, y_train, y_val   = model_selection.train_test_split(X_train, y_train, test_size=dev_split, random_state=1)

     return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    data, lang_labels = avg_data_reader()
    X_train, y_train, X_val, y_val, X_test, y_test = data_split(data, lang_labels)
    
    
    
