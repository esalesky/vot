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

    data  = []
    langs = []
    poa_labels = []
    cross_labels = []
    for k in g.values:
        votcat = k[0]
#        if votcat == 'long.lag':
        poa  = k[1]
        vot  = k[2]
        lang = k[3]
        langs.append(lang)
        data.append(vot)
        poa_labels.append(dpoa[poa])
        label_array = np.zeros(len(cross_dict)) #num_classes
        label_array[cross_dict[(dcat[votcat],dpoa[poa])]] = 1  #one-hot. first dict term: vot category, second term: poa
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
def gmm_train(X_train, y_train, num_classes, covariance='diag', components=3):
    gmm = GMM(n_components=components, n_iter=20, covariance_type=covariance)
    
    X = np.reshape(np.stack(X_train, axis=0), (-1,1)) #reshape to (data_size,1) from (data_size,)
    gmm.means_init = np.array([X[y_train == i].mean(axis=0) #initialize gaussian means to true means
                               for i in range(num_classes)])


    gmm.fit(X)
    print('gmm converged: %s' % gmm.converged_)

    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    return gmm, probs, labels

def gmm_eval(gmm, X_data, labels, set=''):
    X = np.reshape(np.stack(X_data, axis=0), (-1,1))
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
    plt.yticks([])
    plt.scatter(X, len(X)*[0], c=labels, cmap='viridis', s=40, alpha=0.1)

    return

def gaussian_plot(gmm, X, labels, true_labels=True, fig_num=0):
    plt.figure(fig_num)

    means   = gmm.means_.flatten()
    stdevs  = [ np.sqrt(x) for x in gmm.covars_.flatten() ]
    weights = gmm.weights_.flatten()

    x = np.arange(min(X), max(X), 5) #range between data min and max by 5    
    pdfs = [p * ss.norm.pdf(x, mu, sd) for mu, sd, p in zip(means, stdevs, weights)]
    density = np.sum(np.array(pdfs), axis=0)

    #get three colors
    start = 0.0
    stop  = 1.0
    num_lines = len(gmm.covars_) #num_classes
    cm_subsection = np.linspace(start, stop, num_lines)
    colors = [ cm.viridis(x) for x in cm_subsection ]

    #if true_labels map gaussian order to label order
    if true_labels:
        mean_order = np.argsort(means)
        sorted_colors = [colors[i] for i in mean_order]
        color_labels  = [ sorted_colors[i] for i in labels ]
    else:
        color_labels = labels

    #plot gmm
    plt.plot(x, density, 'k--')
    plt.scatter(X, len(X)*[0], c=color_labels, s=40, cmap='viridis', alpha=0.1)

    #plot individual gaussians
    for i, (mu, sd, p) in enumerate(zip(means, stdevs, weights)):
        plt.plot(x, ss.norm.pdf(x, mu, sd), color=colors[i])

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
    num_classes = len(cross_dict)

    data, cross_labels = avg_data_reader(cross_dict, dcat, dpoa)
    X_train, y_train, X_val, y_val, X_test, y_test = data_split(data, cross_labels)
    y_labels = [np.argmax(y) for y in y_train]

    ##train##
    gmm, train_probs, train_predict = gmm_train(X_train, y_labels, num_classes, components=3)

    X = np.reshape(np.stack(X_train, axis=0), (-1,1))
    print(gmm.aic(X))
    print(gmm.bic(X))
    ##eval##
#    xe_train = gmm_eval(gmm, X_train, y_train, 'train')
#    xe_val   = gmm_eval(gmm, X_val, y_val, 'val')
#    xe_test  = gmm_eval(gmm, X_test, y_test, 'test')

    ##plot##
    line_plot(X_train, train_predict, train_probs, fig_num=0)
    line_plot(X_train, y_labels, train_probs, fig_num=1)

    gaussian_plot(gmm, X_train, train_predict, true_labels=False, fig_num=2)
    gaussian_plot(gmm, X_train, y_labels, true_labels=True, fig_num=3)
    plt.show()
    

    print('--done--')
