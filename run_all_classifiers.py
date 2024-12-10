# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def bootstrap(features: pd.array, targets: np.array, size):
    '''
    sampling with replacement in the range (0, n) I will expect n(1 - (1 - 1/n)^k ) unique datapoints. 
    k = n:  n(1 - e^-1) = 0.632n
    k = 2n: n(1 - e^-2) = 0.865n
    k = 3n: n(1 - e^-3) = 0.950n --> 38,000 (30,500 train; 7500 val)
    k = 4n: n(1 - e^-4) = 0.982n
    k = 5n: n(1 - e^-5) = 0.993n --> 5120 (4096 train; 1024 val)

    Return
    ------
    x_train -- pd.ndarray
    x_val   -- pd.ndarray
    y_train -- np.ndarray
    y_val   -- np.ndarray
    '''
    n = len(features)
    all_indices = np.arange(n)
    discard_indices = np.random.choice(all_indices, size=size, replace=True)
    usable_indices = np.setdiff1d(all_indices, discard_indices)
    np.random.shuffle(usable_indices)

    # 80% train, 20% val split
    split = int(0.8 * len(usable_indices))
    train_idx = usable_indices[:split]
    val_idx = usable_indices[split:]
    
    # x_train, y_train, x_val, y_val
    return features.iloc[train_idx], targets[train_idx], features.iloc[val_idx], targets[val_idx]


def preprocess_for_sklearn_tree(X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val, 
                                X_test: pd.DataFrame, y_test, cont_mask):
    '''
    Trims train and val X sets to have all the same categories
        np.unique(X_train) == np.unique(X_val) == np.unique(X_test)
    Converts data to a onehot encoding
    '''

    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse_output=False)

    for col in cont_mask.index:
        if cont_mask[col] == 0:
            
            valid_values = np.intersect1d(
                np.intersect1d(X_train[col].unique(), X_val[col].unique()),
                X_test[col].unique()
            )
            # trim train set to categories that exist in all sets
            y_train = y_train[X_train[col].isin(valid_values)]
            X_train = X_train[X_train[col].isin(valid_values)]
            # trim train set to categories that exist in all sets
            y_val = y_val[X_val[col].isin(valid_values)]
            X_val = X_val[X_val[col].isin(valid_values)]
            # trim train set to categories that exist in all sets
            y_test = y_test[X_test[col].isin(valid_values)]
            X_test = X_test[X_test[col].isin(valid_values)]


    # One hot encode all categorical features
    onehot_xtrain = []
    onehot_xval = []
    onehot_xtest = []

    for col in cont_mask.index:  # Iterate by column index
        if cont_mask[col] == 0:
            # Fit and transform categorical data using OneHotEncoder
            train_col = enc.fit_transform(X_train[col].to_numpy().reshape(-1, 1))
            onehot_xtrain.append(train_col)  # Convert sparse matrix to dense
            val_col = enc.transform(X_val[col].to_numpy().reshape(-1, 1))
            onehot_xval.append(val_col)
            test_col = enc.transform(X_test[col].to_numpy().reshape(-1, 1))
            onehot_xtest.append(test_col)
        else:
            # Append continuous data directly
            onehot_xtrain.append(X_train[col].to_numpy().reshape(-1, 1))
            onehot_xval.append(X_val[col].to_numpy().reshape(-1, 1))
            onehot_xtest.append(X_test[col].to_numpy().reshape(-1, 1))

    # Concatenate the columns to form the final arrays
    X_train = np.concatenate(onehot_xtrain, axis=1)
    X_val = np.concatenate(onehot_xval, axis=1)
    X_test = np.concatenate(onehot_xtest, axis=1)

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    names = [
        "Majority Classifier",
        "Nearest Neighbors",
        # "Linear SVM",
        # "RBF SVM",
        # "Gaussian Process", # Takes too long
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        DummyClassifier(strategy="most_frequent"),
        KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025),
        # SVC(kernel="rbf", gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(n_estimators=10, max_features=1, max_depth=5, bootstrap=False),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(algorithm="SAMME"),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]
    # Hyperparameters
    bootstrap_factor = 3
    bootstrap_trials = 10

    # load datasets
    train_features = pd.read_csv("data/train/predict_recommendation/features.csv")
    train_features_rec = pd.read_csv("data/train/predict_recommendation/targets.csv") # recommendation (binary)
    train_features['recommendation'] = train_features_rec['recommendation']
    test_features = pd.read_csv("data/test/data_about_review.csv")

    # FEATURES
    X_train1 = train_features[
        ["hours_played", "game_name", "publisher", "developer", "overall_player_rating", 
        "helpful", "funny", "n_words", "n_sentences", "polarity", "subjectivity"]]
    X_train2 = train_features[
        ["recommendation", "game_name", "publisher", "developer", "overall_player_rating", 
        "helpful", "funny", "n_words", "n_sentences", "polarity", "subjectivity"]]
    
    X_test1 = test_features[
        ["hours_played", "game_name", "publisher", "developer", "overall_player_rating", 
        "helpful", "funny", "n_words", "n_sentences", "polarity", "subjectivity"]]
    X_test2 = test_features[
        ["recommendation", "game_name", "publisher", "developer", "overall_player_rating", 
        "helpful", "funny", "n_words", "n_sentences", "polarity", "subjectivity"]]
    
    continuous_mask1 = pd.Series([1,0,0,0,0,1,1,1,1,1,1], index=X_train1.columns)
    continuous_mask2 = pd.Series([0,0,0,0,0,1,1,1,1,1,1], index=X_train2.columns)
    
    # OPTIMAL FEATURES
    X_train3 = train_features[['hours_played', 'game_name', 'overall_player_rating']]
    X_test3 = test_features[['hours_played', 'game_name', 'overall_player_rating']]
    continuous_mask3 = pd.Series([0,0,0], index=X_train3.columns)

    # TEXTBLOB
    X_train4 = train_features[
        ["n_words", "n_sentences", "polarity", "subjectivity"]]
    X_test4 = test_features[
        ["n_words", "n_sentences", "polarity", "subjectivity"]]
    continuous_mask4 = pd.Series([1,1,1,1], index=X_train4.columns)

    # TARGETS
    y_train1 = train_features[['hours_played']].to_numpy()
    y_train2 = train_features_rec['recommendation']
    y_test1 = test_features[['hours_played']].to_numpy()
    y_test2 = test_features['recommendation']

    # Discretize hours played
    disc = KBinsDiscretizer(n_bins=8, strategy='quantile', encode='ordinal')
    y_train1 = disc.fit_transform(y_train1).reshape(-1)
    y_test1 = disc.fit_transform(y_test1).reshape(-1)

    # Convert binary recommendation to 0 and 1
    y_train2 = y_train2.map({'Recommended':0, "Not Recommended":1}).to_numpy()
    y_test2 = y_test2.map({'Recommended':0, "Not Recommended":1}).to_numpy()

    dataset_name = [
        # "hours_played", 
        # "recommend",
        "textblob_features",
    ]
    datasets = [
        # predict discretized hours played
        # (X_train2, X_test2, y_train1, y_test1, continuous_mask2),
        # predict recommendation (binary)
        # (X_train1, X_test1, y_train2, y_test2, continuous_mask1),
        # use textblob features to predict recommend
        (X_train4, X_test4, y_train2, y_test2, continuous_mask4)
    ]

    # Init storage .csv
    output = pd.DataFrame(columns=np.arange(bootstrap_trials), index=names)

    for name, clf in zip(names, classifiers):
        for ds_name, ds in zip(dataset_name, datasets):
            for idx in range(bootstrap_trials):

                # split into training and test part
                X, X_ttest, y, y_ttest, cont_mask = ds 

                # perform bootstrapping
                n = len(X)
                X_train, y_train, X_val, y_val = bootstrap(X, y, n*bootstrap_factor)
                X_train, y_train, X_val, y_val, X_test, y_test = preprocess_for_sklearn_tree(
                    X_train, y_train, X_val, y_val, X_ttest, y_ttest, cont_mask)
                
                print("fitting")
                clf = make_pipeline(StandardScaler(), clf)
                clf.fit(X_train, y_train)
                val_score = clf.score(X_val, y_val)
                test_score = clf.score(X_test, y_test)

                print(f"{name}, {idx} ==> val: {val_score}")
                print(f"{name}, {idx} ==> test: {test_score}")
                if name not in output.index:
                    output.loc[name] = [None] * len(output.columns)
                output.loc[name, idx] = str((val_score, test_score))
                output.to_csv(f"all_classifiers/TEXTBLOB_recommend_bs={bootstrap_factor}.csv")

