
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def preprocess_for_sklearn_tree(x_train, y_train, x_val, y_val, cont_mask):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse_output=False)

    # x_train and x_val must contain the same categories.
    # AKA: np.unique(x_train) == np.unique(x_val)
    for col in x_train.columns:
        # only categorical data need to be trimmed
        if cont_mask[col] == 0:
            # Trim the training set to include only categories from the validation set
            y_train = y_train[x_train[col].isin(x_val[col].unique())]
            x_train = x_train[x_train[col].isin(x_val[col].unique())]
            # Trim the training set to include only categories from the validation set
            y_val = y_val[x_val[col].isin(x_train[col].unique())]
            x_val = x_val[x_val[col].isin(x_train[col].unique())]

    # One hot encode all categorical features
    onehot_xtrain = []
    onehot_xval = []

    for col in x_train.columns:
        if cont_mask[col] == 0:
            train_col = enc.fit_transform(x_train[col].to_numpy().reshape(-1,1))
            onehot_xtrain.append(train_col)
            val_col = enc.fit_transform(x_val[col].to_numpy().reshape(-1,1))
            onehot_xval.append(val_col)
        else:
            onehot_xtrain.append(x_train[col].to_numpy().reshape(-1,1))
            onehot_xval.append(x_val[col].to_numpy().reshape(-1,1))

    x_train = np.concatenate(onehot_xtrain, axis=1)
    x_val = np.concatenate(onehot_xval, axis=1)

    return x_train, y_train, x_val, y_val

def RF_model():
    '''
    Feature Subset = Vote count ['recommendation', 'game_name', 'publisher', 'developer', 'overall_player_rating']
    Target Discretization = equal frequencies
    Group size = 8
    Criterion = entropy
    '''
    GROUP = 8
    CRIT = "entropy"

    test_set = pd.read_csv("./data/test/data_about_review.csv")
    test_features = test_set[['recommendation', 'game_name', 'publisher', 'developer', 'overall_player_rating']]
    test_targets = test_set[['hours_played']]

    train_set = pd.read_csv("./data/train/data_about_review.csv")
    train_features = train_set[['recommendation', 'game_name', 'publisher', 'developer', 'overall_player_rating']]
    train_targets = train_set[['hours_played']]

    # convert categorical features to a onehot encoding
    cont_mask = pd.Series([0,0,0,0,0], index=train_features.columns)
    train_features, train_targets, test_features, test_targets = preprocess_for_sklearn_tree(
        train_features, train_targets, test_features, test_targets, cont_mask)

    disc = KBinsDiscretizer(n_bins=GROUP, strategy='quantile', encode='ordinal')
    train_targets = disc.fit_transform(train_targets).reshape(-1)
    test_targets = disc.fit_transform(test_targets).reshape(-1)
    print(test_targets)

    unique, counts = np.unique(train_targets, return_counts=True)
    majority = unique[np.argmax(counts)]
    majority = np.full(len(test_targets), majority)
    print(majority)

    classifier = RandomForestClassifier(n_estimators=100, criterion=CRIT)
    classifier.fit(train_features, train_targets)
    pred = classifier.predict(test_features)
    class_acc = accuracy_score(test_targets, pred)
    majority_acc = accuracy_score(test_targets, majority)
    print(f"Class Acc = {class_acc}")
    print(f"Majority Acc = {majority_acc}")


def KNN_model():
    '''
    Feature Subset = Vote count ['helpful', 'funny']
    Target Discretization = equal frequencies
    Group size = 8
    Value for K = 5 and 1000
    '''
    GROUP = 8
    K = 1000

    test_set = pd.read_csv("./data/test/data_about_review.csv")
    test_features = test_set[['helpful', 'funny']]
    test_targets = test_set[['hours_played']]

    train_set = pd.read_csv("./data/train/data_about_review.csv")
    train_features = train_set[['helpful', 'funny']]
    train_targets = train_set[['hours_played']]

    disc = KBinsDiscretizer(n_bins=GROUP, strategy='quantile', encode='ordinal')
    train_targets = disc.fit_transform(train_targets).reshape(-1)
    test_targets = disc.fit_transform(test_targets).reshape(-1)
    print(train_targets)

    unique, counts = np.unique(train_targets, return_counts=True)
    print(unique, counts)
    majority = unique[np.argmax(counts)]
    majority = np.full(len(test_targets), majority)
    print(majority)

    classifier = KNeighborsClassifier(n_neighbors=K)
    classifier.fit(train_features, train_targets)
    pred = classifier.predict(test_features)
    class_acc = accuracy_score(test_targets, pred)
    majority_acc = accuracy_score(test_targets, majority)
    print(f"Class Acc = {class_acc:.4f}")
    print(f"Majority Acc = {majority_acc:.4f}")


if __name__ == '__main__':
    RF_model()