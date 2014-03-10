def k_fold_eval(X, y, k=5, clf=DecisionTreeClassifier(max_depth=10)):
    '''
    Perform k-fold cross validation on the traind,
    print precision and std deviation and recall and std deviation
    Inputs:
    traind: dictionary in format returned by contest.extract_ibm_data
    k: number of folds
    clf: classifier to use to train the data in each fold
    '''
    kf = cross_validation.KFold(len(X, n_folds=k)

    results = run_folds(kf, clf, X, y)
    print "K-fold XVal Precision and recall for classifier:", clf.__class__
    print "P: %0.2f (+/- %0.2f); R: %0.2f (+/- %0.2f)" \
          % (results[0]/k, results[2], results[1]/k, results[3])


def run_folds(folds, clf, X, y):
    ''' Support method for KFold & stratified KFold
    '''
    tot_precision = 0.0
    tot_recall = 0.0
    p_list = []
    rec_list=[]
    #print "classes", clf.classes_()
    print "num folds:",len(folds)
    for train, test in folds:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        #replace the below two lines with calls to our train/parse machinery
        clf.fit(X_train, y_train)
        labels = clf.predict(X_test)

        #replace the below line with a call to the evaluate method
        pr,rec,_,_ = precision_recall_fscore_support(y_test, labels, pos_label='true')
        #print "p: %f, r: %f" % (pr[1], rec[1])
        tot_precision += pr[1]
        p_list.append(pr[1])
        tot_recall += rec[1]
        rec_list.append(rec[1])
    p_list = np.array(p_list)
    rec_list = np.array(rec_list)
    return tot_precision, tot_recall, np.std(p_list), np.std(rec_list)

