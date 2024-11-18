
import pandas as pd
import numpy as np
from classifiers import KNN
from classifiers import Naive_Bayes
from sklearn.preprocessing import KBinsDiscretizer
import tyro
from dataclasses import dataclass, field


@dataclass
class Args:
    # target_groups: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
    #                                                           55, 60, 65, 70, 75, 80,85,90,95,100])
    target_groups: list[int] = field(default_factory=lambda: [3,8,12,20])
    """Number of groups for target discretization.
        Target discret into equal ranges, equal frequencies, and kmeans clustering."""
    k_choices: list[int] = field(default_factory=lambda: [5, 87])
    """Choices for k. The number of neighbors in the K Nearest Neighbors algorithm."""
    size_distrib: int = 30
    """Number of times to run the classifying algorithm"""
    bootstrap_factor: int = 3
    """Sample with replacement bootstrap_factor * n times"""
    

def bootstrap(features: np.ndarray, targets: pd.Series, size):
    '''
    sampling with replacement in the range (0, n) I will expect n(1 - (1 - 1/n)^k ) unique datapoints. 
    k = n:  n(1 - e^-1) = 0.632
    k = 2n: n(1 - e^-2) = 0.865
    k = 3n: n(1 - e^-3) = 0.950
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
    return features[train_idx], targets[train_idx], features[val_idx], targets[val_idx]


def run_KNN(feature_subset, target):
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
    filename = f"_log\\_knn_trial1\\F={feature_name_list[i]}_T={target_name_list[j]}_GROUP={bin}_K={k}"
    print(filename)

    for idx in range(args.size_distrib): # >=30 samples for theory of large numbers
        # Bootstrap
        n = len(feature_subset)
        x_train, y_train, x_val, y_val = bootstrap(feature_subset, target, size=args.bootstrap_factor*n)

        # Classify
        from sklearn.neighbors import KNeighborsClassifier
        alg = KNeighborsClassifier(
            n_neighbors=k,
            weights="uniform",
        )
        alg.fit(x_train, y_train)
        predictions = alg.predict(x_val)

        # calculate majority class accuracy
        unique, counts = np.unique(y_train, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        majority_acc = (y_val == majority_class).sum() / len(y_val)

        # generate summary statistics
        metrics = gen_metrics(y_val, predictions)
        metrics.insert(0, majority_acc)
        metric_stack.append(np.array(metrics))

    # Print results to a log file
    
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


if __name__ == '__main__':
    args = tyro.cli(Args)

    ############ Create Feature and Target Lists ################

    all_features = np.load("data/train/knn_ready_features.npy")
    
    feature_name_list = ['all', 'vote-count', 'textblob', 'textblob-extra', 'chi-test']
    # feature subset-- numpy array
    all = all_features[:,:]
    vote_count = all_features[:,[0,1]]
    textblob = all_features[:,[2,3]]
    textblob_extra = all_features[:,[2,3,4,5]]
    chi_test = all_features[:,[0,1,3,4,5]]

    feature_list = [all, vote_count, textblob, textblob_extra, chi_test]
    feature_list = [chi_test]
    feature_name_list = ['chi-test']

    targets = pd.read_csv("data/train/knn_ready_target.csv")
    targets = targets.squeeze()

    # Hyperparameter search
    for k in args.k_choices:
        for bin in args.target_groups:
            # Discretize by equal frequencies
            target_freq_disc = pd.qcut(targets, q=bin, labels=np.arange(bin)).to_numpy()

            # Disc by equal range
            target_range_disc = pd.cut(targets, bins=bin, labels=np.arange(bin)).to_numpy()

            # Disc by kmeans clustering
            est = KBinsDiscretizer(n_bins=bin, encode="ordinal", strategy='kmeans')
            est.fit(targets.to_numpy().reshape(-1,1))
            target_kmeans_disc = est.transform(targets.to_numpy().reshape(-1,1)).reshape(-1,).astype(int)

            # Disc by equal range on log(hours_played)
            log_targets = targets.apply(np.log2)
            target_log_range_disc = pd.cut(targets, bins=bin, labels=np.arange(bin)).to_numpy()

            # target_list = [target_range_disc, target_kmeans_disc, target_freq_disc, target_log_range_disc]
            # target_name_list = ['range', 'kmeans', 'freq', 'log_range']
            target_list = [target_log_range_disc]
            target_name_list = ['log_range']

            for j, target in enumerate(target_list):
                for i, feature_subset in enumerate(feature_list):
                    run_KNN(feature_subset, target)