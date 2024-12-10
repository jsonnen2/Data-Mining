
import pandas as pd
import numpy as np
from decision_tree import Decision_Tree
from sklearn.preprocessing import KBinsDiscretizer
import tyro
from dataclasses import dataclass, field


@dataclass
class Args:
    size_distrib: int = 10
    """Number of times to run the classifying algorithm"""
    bootstrap_factor: int = 3
    """Sample with replacement bootstrap_factor * n times"""
    n_estimators: int = 10
    """Number of trees to ensemble in my random forest"""
    criterion: list[str] = field(default_factory=lambda: ["random"])
    """Attribute split strategy"""
    max_features: int = np.Inf
    """Use all available features when making a split decision"""
    bin: list[int] = field(default_factory=lambda: [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    """Number of groups to discretize targets into"""


def bootstrap(features: pd.DataFrame, targets: np.array, size):
    '''
    sampling with replacement in the range (0, n) I will expect n(1 - (1 - 1/n)^k ) unique datapoints. 
    k = n:  n(1 - e^-1) = 0.632n
    k = 2n: n(1 - e^-2) = 0.865n
    k = 3n: n(1 - e^-3) = 0.950n --> 38,000 (30,500 train; 7500 val)
    k = 4n: n(1 - e^-4) = 0.982n
    k = 5n: n(1 - e^-5) = 0.993n --> 5120 (4096 train; 1024 val)
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

def run_my_TREE(feature_subset: pd.DataFrame, target: pd.Series, continuous_mask):
    '''
    Parameters
    ----------
    feature_subset: np.ndarray
        - (n_samples, n_features) -- features to predict on using KNN classifier
    target: np.array
        - (n_samples,) -- discretization of 'hours_played'
    
    1. Bootstrap
        - sampling 3*n points gives a remainder data of ~38,000 --> 30,500 train; 7500 val
    2. Classify
        - K Nearest Neighbors
    3. Majority Class Accuracy and Summary Statistics
    4. Print to _log file
    '''

    metric_list = []
    for idx in range(args.size_distrib): # >=30 samples for theory of large numbers
        # Bootstrap
        print(f"TREE No. {idx}")
        n = len(feature_subset)
        x_train, y_train, x_val, y_val = bootstrap(feature_subset, target, size=args.bootstrap_factor*n)
        y_train = pd.Series(y_train)
        y_val = pd.Series(y_val)

        # Tree 
        alg = Decision_Tree(
            split_type=crit, 
            max_features=args.max_features,
            continuous_mask=continuous_mask
        )
        tree = alg.fit(x_train, y_train)
        predictions = alg.predict(tree, x_val)
        classifier_acc = (y_val == predictions).sum() / len(y_val)
        print(f"Tree Val. Accuracy = {classifier_acc}")

        # calculate majority class on the train data
        unique, counts = np.unique(y_train.to_numpy(), return_counts=True)
        majority_class = unique[np.argmax(counts)]
        # evaluate majority classifier on the validation data
        majority_acc = (y_val == majority_class).sum() / len(y_val)
        print(f"Majority Class Val. Accuracy = {majority_acc: .3f}")

        metric_list.append([majority_acc, classifier_acc])

    # Print results to a log file
    filename = f"P6\\_log_TREE\\F={feature_name_list[i]}_CRIT={crit}"
    column_names = ["Majority Class Accuracy", "Accuracy"]
    summary_stats = pd.DataFrame(metric_list, columns=column_names)
    summary_stats.to_csv(filename, index=False)


def preprocess_for_sklearn_tree(x_train, y_train, x_val, y_val, cont_mask):
    '''
    Trims train and val X sets to have all the same categories
        np.unique(x_train) == np.unique(x_val)
    Converts data to a onehot encoding
    '''

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

def run_sklearn_TREE(feature_subset: pd.DataFrame, target: pd.Series, cont_mask, crit):
    '''
    Parameters
    ----------
    feature_subset: np.ndarray
        - (n_samples, n_features) -- features to predict on using KNN classifier
    target: np.array
        - (n_samples,) -- discretization of 'hours_played'
    
    1. Bootstrap
        - sampling 3*n points gives a remainder data of ~38,000 --> 30,500 train; 7500 val
    2. Classify
        - K Nearest Neighbors
    3. Majority Class Accuracy and Summary Statistics
    4. Print to _log file
    '''
    metric_stack = []
    # filename = f"_log\\_log_TREE\\F={feature_name_list[i]}_T={target_name_list[j]}_GROUP={args.bin}_CRIT={crit}.csv"
    filename = f"./_log/GROUP_hours_1_20/G={group}_F={feature_name_list[i]}_CRIT={crit}.csv"
    print(filename)
    
    for idx in range(args.size_distrib):
        # Bootstrap
        n = len(feature_subset)
        x_train, y_train, x_val, y_val = bootstrap(feature_subset, target, size=args.bootstrap_factor*n)
        x_train, y_train, x_val, y_val = preprocess_for_sklearn_tree(
            x_train, y_train, x_val, y_val, cont_mask)

        crit = "entropy" if crit == "random" else crit
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            criterion=crit,
            max_features=args.max_features,
            max_depth=5,
            bootstrap=False,
        )
        rf.fit(x_train, y_train)
        predictions = rf.predict(x_val)
        accuracy = (y_val == predictions).mean()
        # print(f"Random Forest Accuracy = {accuracy: .3f}")

        # calculate majority class accuracy
        
        unique, counts = np.unique(y_train, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        majority_acc = (y_val == majority_class).sum() / len(y_val)
        # print(f"Majority Class Accuracy = {majority_acc: .3f}")

        # generate summary statistics
        metrics = gen_metrics(y_val, predictions)
        metrics.insert(0, majority_acc)
        metric_stack.append(np.array(metrics))

    column_names = ["Majority Class Accuracy", "Accuracy", "Precision", "Recall", "F Score", "Support"]
    summary_stats = pd.DataFrame(metric_stack, columns=column_names)
    summary_stats.to_csv(filename, index=False)


def gen_metrics(y, yhat):
    '''
    Parameters
    ----------
    y: array-like
        - The true classification labels
    yhat: array-like
        - The predicted classification labels

    Return
    ------
    List[float]
        - List of 4 metrics: accuracy, precision, recall, and F1 score
    '''
    from sklearn import metrics

    classifier_acc = metrics.accuracy_score(y, yhat)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y, yhat, average="weighted", zero_division=0)

    return [classifier_acc, precision, recall, fscore, support]


# if __name__ == '__main__':
def feature_subset_1_pred_rec():
    args = tyro.cli(Args)

    all_features = pd.read_csv("data\\train\\tree_ready_features.csv")
    col_names = ['helpful','funny','polarity','subjectivity','n_words','n_sentences','hours_played',
            'game_name', 'publisher', 'developer', 'overall_player_rating']
    cont_mask = pd.Series([1,1,1,1,1,1,1,0,0,0,0], index=col_names)

    textblob1 = all_features[['n_words']]
    textblob2 = all_features[['n_sentences']]
    textblob3 = all_features[['polarity']]
    textblob4 = all_features[['subjectivity']]

    textblob_length = all_features[['n_words', 'n_sentences']]
    textblob_pair1 = all_features[['n_words', 'polarity']]
    textblob_pair2 = all_features[['n_words', 'subjectivity']]
    textblob_pair3 = all_features[['n_sentences', 'polarity']]
    textblob_pair4 = all_features[['n_sentences', 'subjectivity']]
    textblob_sentiment = all_features[['polarity', 'subjectivity']]

    textblob_all = all_features[['polarity', 'subjectivity', 'n_words', 'n_sentences']]

    feature_name_list = ['textblob1', 'textblob2', 'textblob3', 'textblob4', 
        'textblob-length', 'textblob-sentiment', 'textblob-pair1', 'textblob-pair2', 
        'textblob-pair3', 'textblob-pair4', 'textblob-all']

    feature_list = [
        textblob1, textblob2, textblob3, textblob4, textblob_length, textblob_sentiment, 
        textblob_pair1, textblob_pair2, textblob_pair3, textblob_pair4, textblob_all,
    ]

    targets = pd.read_csv("data/train/predict_recommendation/targets.csv")
    targets = targets.squeeze()

    # Hyperparameter search
    for crit in args.criterion:
        if crit == "random":
            args.max_features = 1
        else:
            args.max_features = None

        for i, features in enumerate(feature_list):
            continuous_mask = cont_mask[features.columns] 
            run_sklearn_TREE(features, targets, continuous_mask, crit)


# if __name__ == '__main__':
def feature_subset_2_pred_rec():
    args = tyro.cli(Args)

    all_features = pd.read_csv("data\\train\\predict_recommendation\\features.csv")

    col_names = ['helpful','funny','polarity','subjectivity','n_words','n_sentences','hours_played',
            'game_name', 'publisher', 'developer', 'overall_player_rating']
    cont_mask = pd.Series([1,1,1,1,1,1,1,0,0,0,0], index=col_names)

    cate1 = all_features[['game_name', 'overall_player_rating']]
    cate2 = all_features[['publisher', 'overall_player_rating']]
    cate3 = all_features[['developer', 'overall_player_rating']]
    cate_all = all_features[['game_name', 'publisher', 'developer', 'overall_player_rating']]

    cate1_recc = all_features[['game_name', 'overall_player_rating', 'hours_played']]
    cate2_recc = all_features[['publisher', 'overall_player_rating', 'hours_played']]
    cate3_recc = all_features[['developer', 'overall_player_rating', 'hours_played']]
    cate_all_recc = all_features[['game_name', 'publisher', 'developer', 'overall_player_rating', 'hours_played']]
    every_feature = all_features[col_names]

    feature_name_list = [  
                         'cate1', 'cate2', 'cate3', 'cate-all', 
                         'cate1-recc', 'cate2-recc', 'cate3-recc', 'cate-all-recc', 'every']
    feature_list = [
                    cate1, cate2, cate3, cate_all,
                    cate1_recc, cate2_recc, cate3_recc, cate_all_recc,
                    every_feature]

    targets = pd.read_csv("data/train/predict_recommendation/targets.csv")
    targets = targets.squeeze()

    # Hyperparameter search
    for crit in args.criterion:
        if crit == "random":
            args.max_features = 1
        else:
            args.max_features = None

        for i, features in enumerate(feature_list):
            continuous_mask = cont_mask[features.columns] 
            run_sklearn_TREE(features, targets, continuous_mask, crit)


if __name__ == '__main__':
# def predict_hours_played():
    args = tyro.cli(Args)

    ############ Create Feature and Target Lists ################

    # 1. Use same feature subsets as KNN
    # 2. Use special decision tree categorical subset

    all_features = pd.read_csv("data\\train\\tree_ready_features.csv")

    col_names = ['helpful','funny','polarity','subjectivity','n_words','n_sentences',
            'game_name', 'publisher', 'developer', 'overall_player_rating', 'recommendation']
    cont_mask = pd.Series([1,1,1,1,1,1,0,0,0,0,0], index=col_names)
  
    # feature subset-- numpy array
    continuous_all = all_features[['helpful', 'funny', 'polarity', 'subjectivity', 'n_words', 'n_sentences']]
    vote_count = all_features[['helpful', 'funny']]
    textblob = all_features[['polarity', 'subjectivity']]
    textblob_extra = all_features[['polarity', 'subjectivity', 'n_words', 'n_sentences']]
    # chi_test = all_features[['helpful', 'funny', 'subjectivity', 'n_words', 'n_sentences']]

    cate1 = all_features[['game_name', 'overall_player_rating']]
    cate2 = all_features[['publisher', 'overall_player_rating']]
    cate3 = all_features[['developer', 'overall_player_rating']]
    cate_all = all_features[['game_name', 'publisher', 'developer', 'overall_player_rating']]

    cate1_recc = all_features[['game_name', 'overall_player_rating', 'recommendation']]
    cate2_recc = all_features[['publisher', 'overall_player_rating', 'recommendation']]
    cate3_recc = all_features[['developer', 'overall_player_rating', 'recommendation']]
    cate_all_recc = all_features[['game_name', 'publisher', 'developer', 'overall_player_rating', 'recommendation']]
    every_feature = all_features[col_names]

    feature_name_list = [#'continuous-all', 'vote-count', 'textblob', 'textblob-extra',  
                         #'cate1', 'cate2', 'cate3', 'cate-all', 
                         #'cate1-recc', 'cate2-recc', 
                         #'cate3-recc', 
                         #'cate-all-recc', ]
                         'every']
    feature_list = [#continuous_all, vote_count, textblob, textblob_extra, 
                    #cate1, cate2, cate3, cate_all,
                    #cate1_recc, cate2_recc, 
                    #cate3_recc, 
                    #cate_all_recc,]
                    every_feature]

    for group in args.bin:

        targets = pd.read_csv("data/train/knn_ready_target.csv")
        targets = targets.squeeze()

        # Discretize by equal frequencies
        target_freq_disc = pd.qcut(targets, q=group, labels=np.arange(group)).to_numpy()

        # Disc by equal range
        target_range_disc = pd.cut(targets, bins=group, labels=np.arange(group)).to_numpy()

        # Disc by kmeans clustering
        est = KBinsDiscretizer(n_bins=group, encode="ordinal", strategy='kmeans')
        est.fit(targets.to_numpy().reshape(-1,1))
        target_kmeans_disc = est.transform(targets.to_numpy().reshape(-1,1)).reshape(-1,).astype(int)

        # Disc by equal range on log(hours_played)
        log_targets = targets.apply(np.log2)
        target_log_range_disc = pd.cut(targets, bins=group, labels=np.arange(group)).to_numpy()

        target_list = [
            # target_kmeans_disc, 
            # target_range_disc, 
            target_freq_disc, 
            # target_log_range_disc,
        ]
        target_name_list = [
            # 'kmeans', 
            # 'range', 
            'freq', 
            # 'log_range',
        ]

        # Hyperparameter search
        for crit in args.criterion:
            if crit == "random":
                args.max_features = 1
            else:
                args.max_features = None

            for j, targets in enumerate(target_list):
                for i, features in enumerate(feature_list):
                    continuous_mask = cont_mask[features.columns] 
                    run_sklearn_TREE(features, targets, continuous_mask, crit)

