import numpy as np
import pandas as pd
import scipy
from scipy.stats import entropy
from sklearn.feature_selection import chi2


class Node():

    def __init__(self, attribute, value, branch, parent, split=None):
        """
        Parameters
        ----------
        attribute : str or None
            The column from `features` to split on at this stage of the tree.
            If this is a leaf node, `attribute` will be set to None.

        branch : dict of str
            A dictionary mapping feature values to the next node. 
            Stores the values as they appear in the data. 
            If len() == 0, this node is a leaf.

        parent : Node or None
            Pointer to the parent node, which allows easier tree traversal.

        value : int
            Stores the majority label, or -1 if below the clash threshold.

        split : float or None
            The value to split on if this is a continuous attribute.
            Set to None for categorical attributes.
        """

        self.attribute = attribute
        self.branch = branch
        self.parent = parent
        self.value = value
        self.split = split


def clash_threshold(targets, clash):
    mode = targets.mode().iloc[0]
    clash = targets.value_counts().get(mode) / len(targets)
    return mode if clash >= clash else -1

def sufficiently_pure(target, epsilon):
    unique, counts = np.unique(target, return_counts=True)
    pmf = counts / len(target)
    E = entropy(pmf, base=2) 
    return E < epsilon

def split_by_random(features: pd.DataFrame, targets: pd.Series) -> str:
    attr_idx = np.random.randint(0, len(features.columns))
    attribute = features.columns[attr_idx]
    return attribute

def splity_by_in_order(features: pd.DataFrame, targets: pd.Series) -> str:
    return features.columns[0]

def split_by_entropy(features: pd.DataFrame, targets: pd.Series) -> str:
    """
    Finds the best attribute to split on using Entropy calculation.

    Parameters
    ----------
    features : pd.DataFrame
        The input features for training, with each row representing a sample and each column a feature.
    targets : pd.Series
        The target values for each sample in the training data.

    Returns
    -------
    split_attribute : str
        The column name to split on.
    """

    min_entropy = np.Inf
    split_attribute = None

    for attribute in features.columns:
        N = len(features)
        entorpy_score = 0.0
        for branch in features[attribute].unique():
            # Find the subset of data that exists in this decision split
            row_indices = features[attribute] == branch
            target_subset = targets[row_indices]
            # Perform entropy calculations
            unique, counts = np.unique(target_subset, return_counts=True)
            pmf = counts / len(target_subset)
            E = entropy(pmf, base=2) 
            fraction = len(target_subset) / N
            entorpy_score += fraction * E
        
        if entorpy_score < min_entropy:
            min_entropy = entorpy_score
            split_attribute = attribute

    return split_attribute

def split_by_chi(features: pd.DataFrame, targets: pd.Series) -> str:
    """
    Finds the best attribute to split on using the Chi-Squared test statistic.

    Parameters
    ----------
    features : pd.DataFrame
        The input features for training, with each row representing a sample and each column a feature.
    targets : pd.Series
        The target values for each sample in the training data.

    Returns
    -------
    split_attribute : str
        The column name to split on.
    """
    attr_idx = np.argmin(chi2(features, targets)[1])
    return features.columns[attr_idx]

def split_by_gain_ratio(features: pd.DataFrame, targets: pd.Series) -> str:
    """
    Finds the best attribute to split on using the Gain Ratio calculation.

    Parameters
    ----------
    features : pd.DataFrame
        The input features for training, with each row representing a sample and each column a feature.
    targets : pd.Series
        The target values for each sample in the training data.

    Returns
    -------
    split_attribute : str
        The column name to split on.
    """
    max_gain_ratio = -np.Inf
    split_attribute = None

    # Counts of each class label
    unique, counts = np.unique(targets, return_counts=True)
    pmf = counts / len(targets)
    # Entropy before splitting
    E_old = entropy(pmf, base=2)


    for attribute in features.columns:
        N = len(features)

        # Entropy across branches (IV)
        unique, counts = np.unique(features[attribute], return_counts=True)
        pmf = counts / len(features)
        IV = entropy(pmf, base=2)

        if IV == 0:
            continue

        # Entropy for each split
        E_new = 0.0
        for branch in unique:
            row_indices = features[attribute] == branch
            target_subset = targets[row_indices] 

            branch_unique, branch_counts = np.unique(target_subset, return_counts=True)
            branch_pmf = branch_counts / len(target_subset)
            branch_entropy = entropy(branch_pmf, base=2) 

            weight = len(target_subset) / N
            E_new += weight * branch_entropy
            
        # Calculate gain ratio: (Information Gain) / (Intrinsic Value)
        # Information Gain = E_old - E_new
        gain_ratio = (E_old - E_new) / IV

        if gain_ratio > max_gain_ratio:
            max_gain_ratio = gain_ratio
            split_attribute = attribute

    return split_attribute


def continuous_mask(features: pd.DataFrame, targets: pd.Series):
    # create a continuous mask
    # an attritube is considered continuous if it is of numeric datatype
    return features.apply(lambda col: pd.api.types.is_numeric_dtype(col))


def discritize_split(features: pd.Series, targets: pd.Series):
    """
    Discretizes a continuous feature series to determine optimal split points for decision making.

    Parameters
    ----------
    features : pd.Series
        The continuous feature values to be discretized.
    targets : pd.Series
        The target values associated with each sample.

    Returns
    -------
    discritized_features : pd.Series
        The same pd.Series of features, except (features <= split) == 0 and (features > split) == 1
        This allows me to toss this data into my attribute selection functions later
    split : float
        The continuous value to split the data on
    """
    # sort features numerically
    sort_idx = features.argsort()
    sorted_features = features.iloc[sort_idx]
    sorted_targets = targets.iloc[sort_idx]

    min_entropy = np.Inf
    split = None

    for i in range(len(sorted_features)-1):
        # I never want to split in between the same datapoint
        if sorted_features.iloc[i] == sorted_features.iloc[i+1]:
            continue

        # perform the split.
        t1 = sorted_targets.iloc[:i+1]
        t2 = sorted_targets.iloc[i+1:]
        #  Calculate entropy
        pmf_t1 = np.unique(t1, return_counts=True)[1] / len(t1)
        pmf_t2 = np.unique(t2, return_counts=True)[1] / len(t2)
        
        E1 = entropy(pmf_t1, base=2)
        E2 = entropy(pmf_t2, base=2)
        # weight entropy by number of targets in each split
        E_new = (1/len(sorted_targets))*(len(t1)*E1 + len(t2)*E2)

        if E_new < min_entropy:
            min_entropy = E_new
            split = sorted_features.iloc[i]

    # All values greater than split are 1. Less than 0.
    # This allows me to toss this data into my split_function()
    discritize_feature = (features > split).astype(int)
    return discritize_feature, split


class Decision_Tree():

    def __init__(self, clash=0.0, epsilon=0.0, split_type="entropy", max_features=None,
                continuous_mask = None):
        '''
        Parameters
        ----------
        clash : float, optional
            used in the clash threshold calculations as a minimum accuracy for the majority class.
        epsilon : float, optional
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

        Returns
        -------
        None

        '''
        self.num_branches = -1 # counter
        self.clash = clash
        self.epsilon = epsilon
        self.split_type = split_type
        self.max_features = max_features
        self.continuous_mask = continuous_mask


    def fit(self, features: pd.DataFrame, targets: pd.Series, parent=None):
        """
        Creates a Decision Tree with the TDIDT algorithm using the provided features and targets.

        Parameters
        ----------
        features : pd.DataFrame
            The input features for training, with each row representing a sample and each column a feature.
        targets : pd.Series
            The target values for each sample in the training data.
        parent : Node, optional
            DO NOT INITIALIZE
            Used during the recursive step. Allows nodes to know their parents.

        Returns
        -------
        tree : Node
            Returns the root node of the constructed tree.
        """
        # 1. Discretize continuous data
        if self.continuous_mask is None:
            self.continuous_mask = continuous_mask(features, targets)

        # for each continuous attribute, find the "best" split
        # "best" determined by minimum entropy
        split_dict = {}
        disc_features = features.copy()
        for attribute in features.columns:
            if self.continuous_mask[attribute]:
                disc_data, split = discritize_split(features[attribute], targets)
                disc_features[attribute] = disc_data
                split_dict[attribute] = split
            else:
                split_dict[attribute] = None

        criterion = {
            "random" : split_by_random,
            "entropy" : split_by_entropy,
            "chi2" : split_by_chi,
            "gain_ratio" : split_by_gain_ratio,
            "in_order" : splity_by_in_order,
        }
        # 2. select a attribute of features to split on
        split_function = criterion.get(self.split_type)
        attribute = split_function(disc_features, targets)
        # Find the value for split associated with attribute
        disc_split = split_dict[attribute]

        # 3. make list of all unique values for that attribute. 
        #    each unique value represents a branch
        branches = disc_features[attribute].unique()

        # 4. make subsets of my data according to the branches
        row_indices = [np.where(disc_features[attribute] == b) for b in branches]
        # row_indices = [disc_features[attribute] == b for b in branches]
        if not self.continuous_mask[attribute]: # only drop the attribute if the data is not continuous
            features = features.drop(attribute, axis=1)
        split_features = [features.iloc[row] for row in row_indices]
        split_targets = [targets.iloc[row] for row in row_indices]

        # 5. resolve clash threshold
        majority_class = clash_threshold(targets, self.clash)

        # 6. create a Node shell. All branches are initialized to point to None.
        #   I am about to loop through all branches to find their proper children
        children = [None] * len(branches)
        tree = Node(attribute, majority_class, dict(zip(branches, children)), parent, disc_split)

        for b, f, t in zip(branches, split_features, split_targets):
            self.num_branches += 1

            if f.empty == True:
                # I have exhausted by features, pick majority class
                majority_class = clash_threshold(t, self.clash)
                leaf = Node(None, majority_class, {}, tree)
                tree.branch[b] = leaf

            elif len(t.unique()) == 1: 
                # all examples in t have the same label
                majority_class = clash_threshold(t, self.clash)
                leaf = Node(None, majority_class, {}, tree)
                tree.branch[b] = leaf

            elif sufficiently_pure(t, self.epsilon):
                # the elements of t are sufficiently similar to predict 1 class label.
                majority_class = clash_threshold(t, self.clash)
                leaf = Node(None, majority_class, {}, tree)
                tree.branch[b] = leaf

            elif np.all(f.to_numpy() == f.to_numpy()[0]):
            # elif f[attribute].eq(f[attribute].iloc[0]).all().all():
                # Adequecy condition not met. All features[attribute] are the same datapoint, but different classes
                # Make tree a leaf node with value of majority class (after checking clash threshold)
                majority_class = clash_threshold(t, self.clash)
                tree.attribute = None
                tree.value = majority_class
                tree.branch = {}
                tree.split = None
                break
                

            else:
                # Recurse
                tree.branch[b] = self.fit(f, t, tree)

        return tree


    def predict(self, tree, features: pd.DataFrame):
        """
        Predicts target values for the given input features using the specified decision tree.

        Parameters
        ----------
        tree : Node
            The trained decision tree model used for making predictions.
        features : pd.DataFrame
            The input features for which predictions are to be made, with each row representing a sample
            and each column a feature.

        Returns
        -------
        pd.Series
            Predicted target values corresponding to each input sample in `features`.
        """
        # init predictions
        predictions = np.full(len(features), tree.value, dtype='U55')
        
        # leaf node
        if len(tree.branch) == 0:
            predictions[:] = [tree.value] * len(predictions)
            return predictions

        # continuous data
        if tree.split != None:
            decision_split = features[tree.attribute]
            # find indices of features that are less than or equal. Then greater than
            less_than_or_equal_indices = np.where(decision_split <= tree.split)[0]
            greater_than_indices = np.where(decision_split > tree.split)[0]
            # Recurse on less than split, then greater than split
            predictions[less_than_or_equal_indices] = self.predict(tree.branch.get(0), features.iloc[less_than_or_equal_indices])
            predictions[greater_than_indices] = self.predict(tree.branch.get(1), features.iloc[greater_than_indices])

        else:
            decision_split = features[tree.attribute].to_numpy()
            for branch, subtree in tree.branch.items():
                match_row_idx = np.where(decision_split == branch)[0]
                predictions[match_row_idx] = self.predict(subtree, features.iloc[match_row_idx])

        return predictions


    def print_tree(self, tree, file: str, indent="\n"):
        """
        Prints a visual representation of the decision tree to the specified file.
        Use sys.stdout to print to console.

        Parameters
        ----------
        tree : Node
            The decision tree model to be printed.
        file : str
            The filepath specified as a string
        indent : str, optional
            DO NOT INITIALIZE
            Use to add indenting for each branch. Lower level branches have more indentation.

        Returns
        -------
        None
            This method prints the tree to the specified file and does not return a value.
        """
        indent = indent + "."

        # leaf node
        if len(tree.branch) == 0:
            print("\t leaf node. class=" + str(tree.value), file=file, end="")
        
        for branch, subtree in tree.branch.items():
            # continuous data
            if tree.split != None:
                if branch == 0:
                    print(f"{indent} ({str(tree.attribute)} <= {str(tree.split)})", file=file, end="")
                elif branch == 1:
                    print(f"{indent} ({str(tree.attribute)} > {str(tree.split)})", file=file, end="")
                else:
                    raise ValueError("A continuous split must be binary")
            else:
                print(indent + str(tree.attribute) + " " + str(branch), file=file, end="")
            self.print_tree(subtree, file, indent)


if __name__ == "__main__":
    print("nothing happens!")