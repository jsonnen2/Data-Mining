
import numpy as np
import pandas as pd
from decision_tree import Decision_Tree


'''
Parameters
    n_estimators: the # of decision trees in the ensemble classifier (default = 100)
    criterion: string denoting the splitting criterion (default = 'entropy')
    max_features: The number of random features to consider when looking for the best split (default = 1)
    node_entropy: float, 0 <= node_entropy <= log2(max_features) Pre-pruning parameter

You will also implement the calculation of the Out-Of-Bag (OOB) estimate for Forest-RI validation performance. 
'''

def bootstrap(features: pd.DataFrame, targets: pd.Series, factor=1):
    '''
    sampling with replacement in the range (0, n) I will expect n(1 - (1 - 1/n)^k ) unique datapoints. 
    k = n:  n(1 - e^-1) = 0.632
    k = 2n: n(1 - e^-2) = 0.865
    k = 3n: n(1 - e^-3) = 0.950
    '''
    n = len(features)
    all_indices = np.arange(n)
    train_indices = np.random.choice(all_indices, size=factor*n, replace=True)
    val_indices = np.setdiff1d(all_indices, train_indices)
    
    # x_train, y_train, x_val, y_val
    return features.iloc[train_indices], targets.iloc[train_indices], features.iloc[val_indices], targets.iloc[val_indices]

class Random_Forest():

    def __init__(self, node_entropy=0.0, n_estimators=100, criterion="entropy", max_features=1):
        '''
        Parameters
        ----------
        node_entropy : float, optional
            If entropy is below this value, predict the majority class.
        split_type : str, optional
            The type of data splitting to be used, by default "random".
            Options include:
            - "in_order" : splis the data in order
            - "random" : split the data randomly.
            - "entropy" : split the data by minimizing entropy.
            - "chi2" : split the data by maximizing chi-squared statistic
            - "gain_ratio" : split the data by maximizing gain-ratio (a variant of entropy)
        max_features : int, optional
            If used, the decision tree will select max_features many features to consider at each step of the decision tree.
        n_estimators : int, optional
            The number of ensembled trees in the Forest.

        Returns
        -------
        None

        '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.node_entropy = node_entropy
        self.forest = []
        self.alg = Decision_Tree(
            split_type = self.criterion,
            max_features = self.max_features,
            epsilon= self.node_entropy
        )

    def fit(self, features: pd.DataFrame, targets: pd.Series):
        """
        Creates an ensembling of Decision Tree which each use the TDIDT algorithm and
          the provided features and targets.

        Parameters
        ----------
        features : pd.DataFrame
            The input features for training, with each row representing a sample and each column a feature.
        targets : pd.Series
            The target values for each sample in the training data.

        Returns
        -------
        validation_accuracy : float
            The accuracy on the bootstrap validation sample
        """
        self.forest = []
        OOB_estimate = pd.DataFrame(
            np.zeros((len(features), self.n_estimators), dtype='U25'), 
            columns=[f"{i+1}" for i in range(self.n_estimators)],
            dtype=str
        )
        # redo indices for f and t
        features = features.reset_index(drop=True)
        targets = targets.reset_index(drop=True)
        
        for i in range(self.n_estimators):
            # bootstrap data
            x_train, y_train, x_val, y_val = bootstrap(features, targets)
            
            tree = self.alg.fit(x_train, y_train)
            self.forest.append(tree)
            OOB_predictions = self.alg.predict(tree, x_val)
            OOB_predictions = pd.Series(OOB_predictions, index=x_val.index)
            OOB_estimate[f"{i+1}"] = OOB_predictions
            
        def most_common_string(row):
            non_nan_values = row.dropna().astype(str)  # Ensure all values are treated as full strings
            mode_result = non_nan_values.mode()
            return mode_result.iloc[0] if not mode_result.empty else None

        # Apply the function along each row and store the result in a Series
        OOB_ensemble = OOB_estimate.apply(most_common_string, axis=1)
        OOB_accuracy = (OOB_ensemble == targets).sum() / len(targets)

        return OOB_accuracy


    def predict(self, xtest, ytest):
        """
        Parameters
        ----------
        xtest : pd.DataFrame
            The input features for which predictions are to be made, with each row representing a sample
            and each column a feature.

        Returns
        -------
        predictions : pd.Series
            Predicted target values corresponding to each input sample in `features`.
        """
        predictions = np.zeros((self.n_estimators, len(xtest)), dtype='<U20')
        for idx, tree in enumerate(self.forest):
            pred = self.alg.predict(tree, xtest)
            predictions[idx] = pred

        ensemble_pred = np.zeros((len(xtest),), dtype='<U20')
        for idx in range(predictions.shape[1]):
            pred = predictions[:,idx]
            unique, counts = np.unique(pred, return_counts=True)
            majority_pred = unique[np.argmax(counts)]
            ensemble_pred[idx] = majority_pred

        test_acc = 0
        for j in range(len(ensemble_pred)):
            if ensemble_pred[j] == ytest.iloc[j]:
                test_acc += 1
        test_acc /= len(ensemble_pred)

        return test_acc

def generate_coding5_output(X, y, xtest, ytest):

    tree_acc = np.zeros((2,10))
    forest_acc = np.zeros((2,10))

    tree_alg = Decision_Tree()
    forest_alg = Random_Forest(n_estimators=100)

    for idx in range(10):
        print(f"Trial = {idx}")
        # Bootstrap
        x_train, y_train, x_val, y_val = bootstrap(X, y)

        # Decision Tree
        decision_tree = tree_alg.fit(x_train, y_train)
        val_pred = tree_alg.predict(decision_tree, x_val)
        val_acc = (val_pred == y_val).sum() / len(y_val)
        test_pred = tree_alg.predict(decision_tree, xtest)
        test_acc = (test_pred == ytest).sum() / len(ytest)

        tree_acc[0,idx] = val_acc
        tree_acc[1,idx] = test_acc

        # Forest
        OOB_acc = forest_alg.fit(X, y)
        test_acc = forest_alg.predict(xtest, ytest)

        forest_acc[0,idx] = OOB_acc
        forest_acc[1,idx] = test_acc

    min_val = np.min(tree_acc[0])
    max_val = np.max(tree_acc[0])
    mean_val = np.mean(tree_acc[0])
    std_val = np.std(tree_acc[0])
    test = np.mean(tree_acc[1])

    tree_output = [min_val, max_val, mean_val, std_val, test]

    min_val = np.min(forest_acc[0])
    max_val = np.max(forest_acc[0])
    mean_val = np.mean(forest_acc[0])
    std_val = np.std(forest_acc[0])
    test = np.mean(forest_acc[1])

    forest_output = [min_val, max_val, mean_val, std_val, test]

    output = np.stack([np.array(tree_output), np.array(forest_output)])
    print(output)
    return output

if __name__ == '__main__':

    # load iris dataset
    iris_data = pd.read_csv("data\\iris.csv")
    iris_test_idx = pd.read_csv("data\\iris_test.txt", header=None)[0] 
    iris_test = iris_data.loc[iris_test_idx]
    iris_train = iris_data.loc[~iris_data.index.isin(iris_test_idx)]
    iris_output = generate_coding5_output(iris_train.iloc[:,:-1], iris_train.iloc[:,-1],
                                          iris_test.iloc[:,:-1], iris_test.iloc[:,-1])

    # load cars dataset
    cars_data = pd.read_csv("data\\cars.csv")
    cars_test_idx = pd.read_csv("data\\cars_test.txt", header=None)[0]
    cars_test = cars_data.loc[cars_test_idx]
    cars_train = cars_data.loc[~cars_data.index.isin(cars_test_idx)]
    cars_output = generate_coding5_output(cars_train.iloc[:,:-1], cars_train.iloc[:,-1],
                                          cars_test.iloc[:,:-1], cars_test.iloc[:,-1])

    with open('rf_results.txt', 'w') as file:
        file.write("Iris output\n")
        np.savetxt(file, iris_output, fmt="%.6f")
        file.write("Cars output\n")
        np.savetxt(file, cars_output, fmt="%.6f")

        file.write("Discussion:")
        file.write("""
            The iris and cars datasets differ in their features types. Iris has continuous data, cars 
            has categorical data. The random forest for both datasets has a much smaller standard error. 
            A random forest is an ensemble of 100 decision trees. The test accuracy of random forest outperformed
            decision tree test accuracy across both datasets. I am glad. Random forest took 100x longer to 
            run. If there were no improvements to test accuracy I would be sad.  """)