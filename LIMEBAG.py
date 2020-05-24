import pandas as pd
import numpy as np
import lime.lime_tabular
import sk as sk



class LIMEBAG():
    def __init__(self,clf, X_train, y_train,X_test,sensitive=None,K=1):
        self.clf=clf
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.sensitive = sensitive
        self.K=K
        self.ranks=None
        self.rankvals=None

        """
        clf: fitted scikit-learn classifier
            according to LIME's doc, clf should have (probability=True) enabled
        X_train: Dataframe
            training data used to build LIME explainer
        y_train: Dataframe or array
            labels of training data
        X_test: Dataframe
            testing data for fairness analysis
        sensitive: 1-D array
            a list of indices of sensitive attribute in X_train
        K: int
           For each instance in X_test's explanation, if any sensitive feature is among the top-K features, send an alert
        """

        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a dataframe")
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test must be a dataframe")
        if isinstance(y_train, pd.DataFrame) or list(y_train):
            pass
        else:
            raise ValueError("y_train must be an series or a dataframe")
        if sensitive and (not isinstance(sensitive, list)) :
            raise ValueError("Sensitive must be either a 1-d array or left as none")
        if (not isinstance(K, int)) or K<0:
            raise ValueError("K must be a non-negative integer")
        elif K>X_train.shape[1]:
            raise ValueError("K cannot be greater than the dimension of attributes")


    def _check(self):
        n_fea = self.X_train.shape[1]
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=self.X_train.values, training_labels=self.y_train,
                                                           feature_names=self.X_train.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification')
        counter = 0
        rank = []
        rankval = []
        total = self.X_test.shape[0]
        for i in range(total):
            ins = explainer.explain_instance(data_row=pd.to_numeric(self.X_test.values[i]),
                                             predict_fn=self.clf.predict_proba, num_features=n_fea, num_samples=5000)
            ind = ins.local_exp[1]
            fair = True
            for j in range(self.K):
                if self.sensitive and ind[j][0] in self.sensitive:
                    counter += 1
                    print('['+str(i)+'/'+str(total)+']',"Unfair!", self.X_test.columns[ind[j][0]], ind[j][0])
                    print('   ', [each[0] for each in ind])
                    fair = False
                    break
            if fair:
                print('['+str(i)+'/'+str(total)+']','Fair!')
                print('   ', [each[0] for each in ind])
            temp = [each[0] for each in ind]
            temp2 = [each[1] for each in ind]
            rank.append(temp)
            rankval.append(temp2)
        return counter, rank, rankval

    def explain(self):
        cnt, cache, cache2 = self._check()
        size = len(cache)
        col_len = self.X_train.shape[1]
        ranks= [[] for n in range(col_len)]
        rankvals = [[] for n in range(col_len)]
        for i in range(size):
            for j in range(len(cache[i])):
                col = cache[i][j]
                ranks[col].append(j)
                rankvals[col].append(np.abs(cache2[i][j]))
        print("Number of unfair instances:", cnt,"out of all",self.X_test.shape[0],"instances")
        self.ranks,self.rankvals = ranks,rankvals
        return ranks, rankvals

    def find_rank(self,type = 'values',higher=False,latex=False):
        if (not self.rankvals) or (not self.ranks):
            raise ValueError("Must call the explain() function first")
        cols = self.X_test.columns
        col_len = len(cols)
        if type == 'ranks':
            f = open("lime_rank" + ".txt", "w")
            for j in range(col_len):
                f.write(cols[j] + '\n')
                for each in self.ranks[j]:
                    f.write("%f" % each + ' ')
                if j!=col_len-1:
                    f.write('\n')
            f.close()
            f1=open('lime_rank.txt', 'r')
            sk.main(file=f1,higher=higher,latex=latex)
        elif type =='values':
            f = open("lime_val" + ".txt", "w")
            for j in range(col_len):
                f.write(cols[j] + '\n')
                for each in self.rankvals[j]:
                    f.write("%f" % np.round(np.abs(each),4) + ' ')
                if j != col_len - 1:
                    f.write('\n')
            f.close()
            f1 = open('lime_val.txt', 'r')
            sk.main(file=f1,higher=higher,latex=latex)
        else:
            raise ValueError("Expected type to be either values or ranks.")
        return 0