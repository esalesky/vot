from vot import *


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
    gmm, train_probs, train_predict = gmm_train(X_train, y_labels, num_classes, model=BayesianGaussianMixture, covariance='diag', components=9, n_iter=150)
        
    ##eval##
    X = np.reshape(np.stack(X_train, axis=0), (-1,1))
#    print('AIC: %f' % gmm.aic(X)) #BayesianGaussianMixture has no aic, bic
#    print('BIC: %f' % gmm.bic(X))
#    xe_train = gmm_eval(gmm, X_train, y_train, 'train')
#    xe_val   = gmm_eval(gmm, X_val, y_val, 'val')
#    xe_test  = gmm_eval(gmm, X_test, y_test, 'test')

    ##plot##
    gaussian_plot(gmm, X_train, train_predict, true_labels=False, fig_num=1)

    plt.show()
    

    print('--done--')
