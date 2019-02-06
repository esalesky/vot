# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.mixture import GMM
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns; sns.set()
import scipy.stats as ss
from itertools import product

### Data Handling ###
def avg_data_reader(cross_dict, dcat, dpoa, fname='data/ChodroffGoldenWilson2019_vot_avg.csv'):
    """reads in data from ChodroffGoldenWilson2019_vot_avg.csv,
    expected format: family,language,dialect,vot.category,poa2,vot.mu

    discards languages without all 3 poa per vot.category"""

    full = pd.read_csv('data/ChodroffGoldenWilson2019_vot_avg.csv', delimiter = ',', na_filter=False)            
    full['lang'] = full[['family', 'language', 'dialect']].apply(lambda x: '_'.join(x), axis=1)
    full = full.drop(['family', 'language', 'dialect'], axis=1)
    g = full.groupby(['lang','vot.category']).filter(lambda x: len(x) == 3) #drops languages w/o all 3 poa per vot.cat
    X = g.groupby(['lang','vot.category'])['vot.mu'].apply(list)

    data  = []
    langs = []
    votcs = []
    cross_labels = []
    for k in X.keys():
        lang = k[0]
        votcat = dcat[k[1]]
        c,d,l = X[k] #IMPT: coronal, dorsal, labial (sorted by alpha name after groupby) 
        langs.append(k[0])
        data.append([l,c,d])
        votcs.append(votcat)
        label_array = np.zeros(len(dcat)) #num cats
        label_array[votcat] = 1 #one-hot. first dict term: vot category
        cross_labels.append(label_array)

    return data, cross_labels

#def raw_data_reader(fname='data/ChodroffGoldenWilson2019_vot.csv'):
#    """reads in data from ChodroffGoldenWilson2019_vot.csv,
#    expected format: family,language,dialect,source,primarySource,poa1,poa2,vot.category,vot,voiceContrast,notes
#    """

def data_split(X, y, train_split=0.8, dev_split=0.1, test_split=0.1):
    """splits data into train, dev, test sets"""
    assert train_split+dev_split+test_split == 1, "data splits do not add up to 1"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=1, stratify=y)
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=dev_split, random_state=1, stratify=y)

    return X_train, y_train, X_val, y_val, X_test, y_test


### GMM Modeling ###
def gmm_train(X_train, y_train, num_gaussians, num_vars, covariance='diag', components=1):
    gmm = GMM(n_components=components, n_iter=20, covariance_type=covariance)

    X = np.reshape(np.stack(X_train, axis=0), (-1,num_vars)) #reshape to (data_size,1) from (data_size,)
    gmm.means_init = np.array([X[y_train == i].mean(axis=0) #initialize gaussian means to true means
                               for i in range(num_gaussians)])


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


### Plotting ###
def line_plot(X, labels, probs, fig_num=0):
    plt.figure(fig_num)
    #todo: vert distrib classes slightly for clearer view
#    plt.scatter(X, len(X)*[1], c=labels, s=40, cmap='viridis')
    size = 50 * probs.max(1) ** 2  # square emphasizes differences
    print("--")
    print(len(X))
    print(len(labels))
    print(X[0:10])
    print(labels[0:10])
    plt.scatter(X, labels, c=labels, cmap='viridis', s=size, alpha=0.1)

    return

#TODO: figure out how to display this properly. marginalize out one poa..? need all features to estimate probability
def gaussian_plot(gmm, X, labels, fig_num=0):
    plt.figure(fig_num)
    
    means   = gmm.means_.flatten()
    covars  = gmm.covars_[0]
    print(means)
    print(covars)
    
    y = ss.multivariate_normal.pdf(X, means, covars)

    plt.plot(X, y, 'k--')
    print(y[0:10])
#    plt.scatter(X_flat, len(X_flat)*[0], c=labels_flat, s=40, cmap='viridis')
    
    start = 0.0
    stop  = 1.0
    num_lines = len(gmm.covars_) #num_classes

    cm_subsection = np.linspace(start, stop, num_lines)
    colors = [ cm.viridis(x) for x in cm_subsection ]

    return


### Main ###   
if __name__ == '__main__':
    ##data prep##
    #todo: make this easier to modify to test other class groupings
    dcat = {'lead':0, 'short.lag':1, 'long.lag':2}
    dpoa = {'labial':0, 'coronal':0, 'dorsal':0}
#    cross_dict = {(0,0):0,(0,1):1,(0,2):2,(1,0):3,(1,1):4,(1,2):5,(2,0):6,(2,1):7,(2,2):8}
    cross_dict = {}
    for idx, p in enumerate(product(set(dcat.values()), set(dpoa.values()))):
        cross_dict[p] = idx
    num_gaussians = len(dcat) #TODO: make this the number of unique labels

    data, cross_labels = avg_data_reader(cross_dict, dcat, dpoa)
    X_train, y_train, X_val, y_val, X_test, y_test = data_split(data, cross_labels)
    y_labels = [np.argmax(y) for y in y_train]

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_labels = np.array(y_labels)
    
    ##train##
    gmm, train_probs, train_predict = gmm_train(X_train, y_labels, num_gaussians, num_gaussians, covariance='diag') #TODO fix ( num gaussians, num variables)

    print(gmm.aic(X_train))
    print(gmm.bic(X_train))

    ##eval##
#    xe_train = gmm_eval(gmm, X_train, y_train, 'train')
#    xe_val   = gmm_eval(gmm, X_val, y_val, 'val')
#    xe_test  = gmm_eval(gmm, X_test, y_test, 'test')

    ##plot##
    gaussian_plot(gmm, X_train, train_predict, fig_num=1) #note: train_predict will of course have 1 class (num_components=1)

    plt.show()
    

    print('--done--')
