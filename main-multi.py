# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns; sns.set()
import scipy.stats as ss
from itertools import product

### Data Handling ###
def avg_data_reader(fname='data/ChodroffGoldenWilson2019_vot_avg.csv'):
    """reads in data from ChodroffGoldenWilson2019_vot_avg.csv,
    expected format: family,language,dialect,vot.category,poa2,vot.mu

    discards languages without all 3 poa per vot.category"""

    full = pd.read_csv('data/ChodroffGoldenWilson2019_vot_avg.csv', delimiter = ',', na_filter=False)
    full['lang'] = full[['family', 'language', 'dialect']].apply(lambda x: '_'.join(x), axis=1)
    df = full.drop(['family', 'language', 'dialect'], axis=1)
    df = full.groupby(['lang','vot.category']).filter(lambda x: len(x) == 3) #drops languages w/o all 3 poa per vot.cat
    
    all_langs = df['lang'].unique()
    cats  = df['vot.category'].unique()

    langs = []
    data  = []
    for ll in all_langs:
        #IMPT: coronal, dorsal, labial (sorted by alpha name after groupby)
        #Reminder: short=unaspirated,long=aspirated,lead=voiced
        short = list(df.loc[(df['lang'] == ll) & (df['vot.category'] == 'short.lag')]['vot.mu'])
        long  = list(df.loc[(df['lang'] == ll) & (df['vot.category'] == 'long.lag')]['vot.mu'])
        lead  = list(df.loc[(df['lang'] == ll) & (df['vot.category'] == 'lead')]['vot.mu'])
    
        if short and lead:
            langs.append(ll)
            sc,sd,sl = short
            ec,ed,el = lead
            collect = [sl,sc,sd,el,ec,ed]
            data.append(collect)
        
    return data, langs

#def raw_data_reader(fname='data/ChodroffGoldenWilson2019_vot.csv'):
#    """reads in data from ChodroffGoldenWilson2019_vot.csv,
#    expected format: family,language,dialect,source,primarySource,poa1,poa2,vot.category,vot,voiceContrast,notes
#    """

def data_split(X, y, train_split=0.8, dev_split=0.1, test_split=0.1):
    """splits data into train, dev, test sets"""
    assert train_split+dev_split+test_split == 1, "data splits do not add up to 1"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=1) #todo: set stratify for both uni and multi
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=dev_split, random_state=1)

    return X_train, y_train, X_val, y_val, X_test, y_test


### GMM Modeling ###
def gmm_train(X_train, num_vars, model=GaussianMixture, covariance='diag', components=1):
    gmm = model(n_components=components, max_iter=20, covariance_type=covariance)
    print('-- Model: %s, cov: %s --' % (model.__name__, covariance))

    X = np.reshape(np.stack(X_train, axis=0), (-1,num_vars)) #reshape to (data_size,num_vars) from e.g. list of (data_size,)

    gmm.fit(X)
    print('gmm converged: %s' % gmm.converged_)

    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    return gmm, probs, labels

def gmm_eval(gmm, X, labels, set=''):
#    X = np.reshape(np.stack(X_data, axis=0), (-1,1)) #TODO fix
    probs = gmm.predict_proba(X)

    xe = 0.
    xe_all = []
    for row, (gprob, lprob) in enumerate(zip(probs, labels)):
        xe_ent = 0.
        for idx in range(len(gprob)):
            xe_ent -= lprob[idx] * np.log(gprob[idx])
        xe_all.append(xe_ent)
    xe = sum(xe_all)/len(xe_all)
    print('%s xe: %.3f' % (set, xe))

    return xe

def gmm_score(gmm, data, set=''):
    avglogprob = gmm.score(data)
    print('%s avg logprob: %.3f' % (set, avglogprob))

    return avglogprob


### Plotting ###
def line_plot(X, labels, probs, fig_num=0):
    plt.figure(fig_num)
    #todo: vert distrib classes slightly for clearer view
    plt.scatter(X, labels, c=labels, cmap='viridis', s=size, alpha=0.1)

    return

def lang_likelihood_plot(gmm, X, labels, fig_num=0):
    plt.figure(fig_num)
    
#    plt.plot(X, y, 'k--')
    
    return


### Main ###   
if __name__ == '__main__':
    ##data prep##
    #todo: make this easier to modify to test other class groupings
    
    data, lang_labels = avg_data_reader()
    num_features = len(data[0])

    X_train, y_train, X_val, y_val, X_test, y_test = data_split(data, lang_labels)
    X_train = np.array(X_train)
    
    ##train##
    gmm, train_probs, train_predict = gmm_train(X_train, num_features, model=GaussianMixture, components=1, covariance='full') 

    ##eval##
    gmm_score(gmm, X_train, 'train')
    gmm_score(gmm, X_val, '  val')
    gmm_score(gmm, X_test, ' test')

    print('AIC: %f' % gmm.aic(X_train))
    print('BIC: %f' % gmm.bic(X_train))

    ##plot##
#    lang_likelihood_plot(gmm, X, labels, fig_num=1)


    plt.show()
    

    print('--done--')
