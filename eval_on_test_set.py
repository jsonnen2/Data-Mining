
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def preprocess_for_sklearn_tree(x_train, y_train, x_val, y_val, cont_mask):

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
    enc = OneHotEncoder(sparse_output=False)

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


# if __name__ == '__main__':
def MLP_predict_hours():
    '''
    Feature Subset = Vote count ['helpful', 'funny']
    Target Discretization = equal frequencies
    Group size = 8
    Value for K = 5 and 1000
    '''

    # load data
    test_set = pd.read_csv("./data/test/data_about_review.csv")
    test_features = test_set[[
        "recommendation", "game_name", "publisher", "developer", "overall_player_rating", 
        "helpful", "funny", "n_words", "n_sentences", "polarity", "subjectivity"]]
    test_targets = test_set[['hours_played']]

    train_set = pd.read_csv("./data/train/data_about_review.csv")
    train_features = train_set[[
        "recommendation", "game_name", "publisher", "developer", "overall_player_rating", 
        "helpful", "funny", "n_words", "n_sentences", "polarity", "subjectivity"]]
    train_targets = train_set[['hours_played']]

    # convert categorical features to a onehot encoding
    cont_mask = pd.Series([0,0,0,0,0,1,1,1,1,1,1], index=train_features.columns)
    train_features, train_targets, test_features, test_targets = preprocess_for_sklearn_tree(
        train_features, train_targets, test_features, test_targets, cont_mask)
    
    # discretize targets
    disc = KBinsDiscretizer(n_bins=8, strategy='quantile', encode='ordinal')
    train_targets = disc.fit_transform(train_targets).reshape(-1)
    test_targets = disc.fit_transform(test_targets).reshape(-1)

    # Majority Classifier
    unique, counts = np.unique(train_targets, return_counts=True)
    majority = unique[np.argmax(counts)]
    majority = np.full(len(test_targets), majority)
    majority_acc = accuracy_score(test_targets, majority)
    print(f"Majority Acc = {majority_acc:.4f}")

    # MLP Classifier
    classifier = MLPClassifier(alpha=1, max_iter=1000)
    classifier.fit(train_features, train_targets)
    pred = classifier.predict(test_features)
    class_acc = accuracy_score(test_targets, pred)
    print(f"Class Acc = {class_acc:.4f}")

    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(test_targets, pred)
    print(conf_matrix)
    np.savetxt("confusion_matrices/_MLP_hours_best_classifier.csv", conf_matrix, delimiter=',')

if __name__ == '__main__':
    # change confusion matrices
    filename = "confusion_matrices/_RF_recommend_best_classifier_trees=100"
    labels = ["Not Recommend", "Recommend"]

    conf_matrix = np.loadtxt(f"{filename}.csv", delimiter=',').astype(int)
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,6), dpi=100)
    display = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    ax.set(title='Confusion Matrix Random Forest classifier')
    display.plot(ax=ax)
    plt.yticks(rotation=90)
    plt.savefig(f"{filename}.png")
    plt.show()

# if __name__ == '__main__':
def RF_predict_recommend():

    # load data
    test_set = pd.read_csv("./data/test/data_about_review.csv")
    test_features = test_set[[
        "hours_played", "game_name", "publisher", "developer", "overall_player_rating", 
        "helpful", "funny", "n_words", "n_sentences", "polarity", "subjectivity"]]
    test_targets = test_set[['recommendation']]

    train_set = pd.read_csv("./data/train/data_about_review.csv")
    train_features = train_set[[
        "hours_played", "game_name", "publisher", "developer", "overall_player_rating", 
        "helpful", "funny", "n_words", "n_sentences", "polarity", "subjectivity"]]
    train_targets = train_set[['recommendation']]

    # convert categorical features to a onehot encoding
    cont_mask = pd.Series([1,0,0,0,0,1,1,1,1,1,1], index=train_features.columns)
    train_features, train_targets, test_features, test_targets = preprocess_for_sklearn_tree(
        train_features, train_targets, test_features, test_targets, cont_mask)

    # get Majority Class accuracy
    unique, counts = np.unique(train_targets, return_counts=True)
    majority = unique[np.argmax(counts)]
    majority = np.full(len(test_targets), majority)
    majority_acc = accuracy_score(test_targets, majority)
    print(f"Majority Acc = {majority_acc: .4f}")

    # fit random forest
    classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", max_features=1, bootstrap=True) # CRIT = random
    classifier.fit(train_features, train_targets.squeeze())
    pred = classifier.predict(test_features)
    class_acc = accuracy_score(test_targets, pred)
    print(f"Class Acc = {class_acc: .4f}")

    print(type(pred))
    print(np.unique(pred))
    

    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(test_targets, pred)
    print(conf_matrix)
    np.savetxt("confusion_matrices/_RF_recommend_best_classifier_100.csv", conf_matrix, delimiter=',')

