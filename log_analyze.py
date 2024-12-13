
import pandas as pd

# TODO: local_dir as an argparse argument
# TODO: calculate standardized t-score as the first column

def parse_filename_column(local_dir):
    '''
    - All data is from equal frequency target disc strategy.
    - combine sample distributions of same type feature subset
    - perform pairwise comparisons to test if any mean is stat signif different from another
        - H0: mu1 - mu2 = 0
        - Ha: two-tailed
    '''
    data = pd.read_csv(f"{local_dir}_summary_file.csv")

    data['components'] = data['filename'].str.split("_")
    variables = {}

    # Iterate through the components and extract the key-value pairs
    parsed_df = []
    for row in data['components']:
        parsed_row = pd.Series(index=['F','T','GROUP','K'], dtype=str)
        for component in row:
            if '=' in component:
                key, value = component.split('=')
                parsed_row[key] = value
        parsed_df.append(parsed_row)
        
    parsed_df = pd.DataFrame(parsed_df)
    data = pd.concat([data, parsed_df], axis=1)
    data.to_csv(f"{local_dir}_summary_file.csv", index=False)
    return data

if __name__ == '__main__':
    '''
    analyze_log output:
        label -- (same as filename)
        result of walsh t-test for unequal variance 
            - between classifier and majority class         
        classifier_mean
        majority_class_mean
        classifier_variance
        majority_class_variance
        classifier 95% CI
        majority_class 95% CI

    '''

    import os
    local_dir = "./_log/_recommend_9_hopeful" 

    column_names = [
        "filename",
        "t_score",
        "p_value",
        "stat_signif",
        "classifier_mean", "majority_class_mean", 
        "classifier_std_error", "majority_class_std_error",
        "classifier_CI_str", "majority_class_CI_str",
    ]
    results_df = pd.DataFrame(columns=column_names)

    for filename in os.listdir(local_dir):

        components = filename.split("_")
        variables = {}
        
        for component in components:
            # Split each component by '=' to separate keys and values
            if '=' in component:
                key, value = component.split("=")
                variables[key] = value

        # Extract specific variables
        F = variables.get("F")
        T = variables.get("T")
        GROUP = variables.get("GROUP")
        K = variables.get("K")

        # load dataframe
        file_path = os.path.join(local_dir, filename)
        data = pd.read_csv(file_path)

        # calculate classifier confidence interval
        classifier_accuracy = data['Accuracy']
        classifier_mean = classifier_accuracy.mean()
        classifier_std_error = classifier_accuracy.std()
        margin_of_error = 1.96 * classifier_std_error
        lower_bound = classifier_mean - margin_of_error
        upper_bound = classifier_mean + margin_of_error
        classifier_CI_str = f"[{lower_bound:.4f}, {upper_bound:.4f}]"

        # calculate majority class confidence interval
        majority_class_accuracy = data['Majority Class Accuracy']
        majority_class_mean = majority_class_accuracy.mean()
        majority_class_std_error = majority_class_accuracy.std()
        margin_of_error = 1.96 * majority_class_std_error
        lower_bound = majority_class_mean - margin_of_error
        upper_bound = majority_class_mean + margin_of_error
        majority_class_CI_str = f"[{lower_bound:.4f}, {upper_bound:.4f}]"

        from scipy.stats import ttest_ind
        t_test_result = ttest_ind(classifier_accuracy, majority_class_accuracy, equal_var=False)
        if t_test_result.pvalue < 0.05:
            if (classifier_accuracy.mean() > majority_class_accuracy.mean()):
                stat_signif = 1
            else:
                stat_signif = -1
        else:
            stat_signif = 0

        result = [filename, 
                  t_test_result.statistic,
                  t_test_result.pvalue,
                  stat_signif,
                  classifier_mean, majority_class_mean, 
                  classifier_std_error, majority_class_std_error,
                  classifier_CI_str, majority_class_CI_str]
        
        results_df = pd.concat([results_df, pd.DataFrame([result], columns=column_names)], ignore_index=True)


    filename = f"{local_dir}_summary_file.csv"
    results_df.to_csv(filename, index=False)
    parse_filename_column(local_dir)

