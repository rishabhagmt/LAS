import warnings
import shap
import pandas as pd
warnings.filterwarnings('ignore')

class SHAP():
    def __init__(self, clf, X_train, X_test, n_cluster = 10):
        self.clf = clf
        self.X_train = X_train
        self.X_test = X_test
        self.n_cluster = n_cluster
        self.shap=None

        """
        This class is a brief wrap-up of a existing package shap.
        
        clf: fitted scikit-learn classifier
                according to LIME's doc, clf should have (probability=True) enabled
        X_train: Dataframe
                training data used to build LIME explainer
        X_test: Dataframe
                testing data for fairness analysis
        """

        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a dataframe")
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test must be a dataframe")
        if (not isinstance(n_cluster, int)) or n_cluster<1:
            raise ValueError("K must be a positive integer")

    def explain(self):
        shap.initjs()
        X_train_summary = shap.kmeans(self.X_train, self.n_cluster)
        rf_explainer = shap.KernelExplainer(self.clf.predict, X_train_summary)
        shap_values_RF_test = rf_explainer.shap_values(self.X_test)
        self.shap=shap_values_RF_test
        return shap_values_RF_test

    def summary_plot(self):
        if not list(self.shap):
            raise ValueError("Must call the explain() function first")
        return shap.summary_plot(self.shap, self.X_test)