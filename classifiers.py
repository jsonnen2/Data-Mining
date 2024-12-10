import pandas as pd
import numpy as np


def calc_dist(x1, x2, distance_metric):
    """
    Calculate the distance between two points using the specified metric.
    
    Args:
        x1: Feature matrix
        x2: Point to calculate distance to
        distance_metric (str): Distance metric to use ('euclid', 'cityblock', or 'hamming')
    
    Returns:
        Distance between x1 and x2 using the specified metric
        
    Raises:
        ValueError: If the distance metric is not supported
    """
    if distance_metric == "euclid":
        return euclid(x1, x2)
    elif distance_metric == "cityblock":
        return cityblock(x1, x2)
    elif distance_metric == "hamming":
        return hamming(x1, x2)
    else:
        raise ValueError(f"{distance_metric} not a valid distance metric.")

def hamming(features, z):
    # for each feature, count the number of same datapoints
    idx = np.where(features == z)[0]
    unique, counts = np.unique(idx, return_counts=True)
    result = np.zeros(features.shape[0])
    result[unique] = counts # TODO: sus!
    return features.shape[1] - result

def euclid(features, z):
    return np.sqrt(np.sum( (features - z) ** 2, axis=1 ))

def cityblock(features, z):
    return np.sum(np.abs(features - z), axis=1)

class KNN():
    def fit(self, features: np.array, targets: np.array):
        """
        Fit the K-Nearest Neighbors classifier with training data.
        
        Args:
            features: Feature matrix of training data (numpy matrix)
                     Should contain continuous data (or categorical if using Hamming distance)
            targets: Target labels for training data (pandas Series)
                    The order must correspond to the order of features
                    
        Returns:
            None. Stores the training data in the instance.
        """
        self.features = features
        self.targets = targets

    def predict(self, xtest: np.array, distance_metric="euclid", k=5):
        """
        Predict class labels for test data using K-Nearest Neighbors.
        
        Args:
            xtest: Feature matrix to predict labels for
            dist (str): Distance metric to use ('euclid', 'cityblock', or 'hamming')
            k (int): Number of nearest neighbors to use for prediction
            
        Returns:
            list: Predicted class labels for test data
        """
        predictions = []
        for idx, x in enumerate(xtest):
            distance = calc_dist(self.features, x, distance_metric)
            k_nearest = np.argsort(distance)[:k]
            unique, count = np.unique(self.targets[k_nearest], return_counts=True)
            pred = unique[np.argmax(count)]
            predictions.append(pred)

            # if (idx + 1) % 1000 == 0:
            #     print(f"[{idx+1} / {len(xtest)}]")
        return np.array(predictions)


class Naive_Bayes():
    def fit(self, features: pd.DataFrame, targets: np.ndarray) -> None:
        """
        Fit the Naive Bayes classifier with training data.
        
        Args:
            features: Training feature matrix (pandas DataFrame)
                     Should contain categorical data
            targets: Target labels for training data (numpy array)
                    Should contain categorical class labels
                    
        Returns:
            None. Stores the Bayes probability tables in the instance.
        """
        bayes_table = {}
        for attribute in features.columns:
            freq_counts = pd.DataFrame(columns= list(set(targets)))
            for x, y in zip(features[attribute], targets): # upset with this loop!
                if x not in freq_counts.index:
                    freq_counts.loc[x] = [0] * len(freq_counts.columns)
                freq_counts.loc[x, y] += 1

            bayes_table[attribute] = freq_counts
        self.bayes_table = bayes_table

        # keeps order consistant with bayes_table class order
        unique, counts = np.unique(targets, return_counts=True)
        mapping = dict(zip(unique, counts))
        class_probs = [mapping[s] for s in list(set(targets))]
        self.class_probs = class_probs / np.sum(class_probs)


    def predict(self, features: pd.DataFrame):
        """
        Predict class labels for test data using Naive Bayes.
        
        Args:
            features: Test feature matrix to predict labels for (pandas DataFrame)
                     Should contain categorical data matching the training data format
                     
        Returns:
            numpy.ndarray: Predicted class labels for test data
        """
        log_probs = None
        for attr in features.columns:
            table = self.bayes_table[attr]
            # just the counts right now
            conditional_probs = table.loc[features[attr]].to_numpy().astype(float)
            # convert to probabilities
            class_sums = np.sum(table.to_numpy(), axis=0)
            conditional_probs /= class_sums

            if log_probs is None:
                log_probs = np.zeros_like(conditional_probs)
            log_probs += np.log(conditional_probs + 1e-10)
        log_probs *= np.log(self.class_probs)

        class_labels = np.array(self.bayes_table[features.columns[0]].columns)
        predictions = np.array(class_labels[np.argmin(log_probs, axis=1)])
        return predictions