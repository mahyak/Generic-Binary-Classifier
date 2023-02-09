from typing import List, Callable
import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass

FeatureImportanceTechnique = Callable[[pandas.DataFrame], List[bool]]

def pearson_correlation_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()

    for i in X.columns.tolist():
        cor = numpy.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)

    cor_list = [0 if numpy.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:,numpy.argsort(numpy.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]

    return cor_support

def chi_square_selector(X, y,num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return chi_support


def rfe_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(estimator=LogisticRegression(max_iter=1200000), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    return rfe_support


def lasso_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2", max_iter=1200000), max_features=num_feats)
    embeded_lr_selector.fit(X_norm, y)
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
    return embeded_lr_support

def tree_based_selector(X, y, num_feats):
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embeded_rf_selector.fit(X, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    return embeded_rf_support

@dataclass
class FeatureImportance:
    X: pandas.DataFrame
    y: pandas.DataFrame
    num_feats: int

    def apply_selector(self, selector: FeatureImportanceTechnique):
        return selector(self.X, self.y, self.num_feats)
    
def apply_feature_importance(X: pandas.DataFrame, y: pandas.DataFrame, num_feats: int):
    feature_importance = FeatureImportance(X, y, num_feats)
    columns = X.columns

    pearson_support = feature_importance.apply_selector(pearson_correlation_selector)  
    chi_support = feature_importance.apply_selector(chi_square_selector)  
    rfe_support = feature_importance.apply_selector(rfe_selector)  
    embeded_lr_support= feature_importance.apply_selector(lasso_selector)
    embeded_rf_support = feature_importance.apply_selector(tree_based_selector)

    feature_selection_df = pandas.DataFrame({'Feature': columns, 'Pearson':pearson_support, 'Chi-2':chi_support, 
        'RFE':rfe_support, 'Logistics':embeded_lr_support,'Random Forest':embeded_rf_support})
                                    
    feature_selection_df['Total'] = numpy.sum(feature_selection_df, axis=1)

    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    return feature_selection_df

    
