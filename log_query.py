
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.lines as mlines

def analysis_of_textblob_feature_subsets():
    data1 = pd.read_csv("./_log/_log_rec_textblob_summary_file.csv")
    data2 = pd.read_csv("./_log/_log_rec_8_hopeful_summary_file.csv")
    data = pd.concat([data1, data2], axis=0)
    data = data[data['stat_signif'] == 1]
    t = data.groupby('CRIT')['t_score'].mean()
    p = data.groupby('CRIT')['p_value'].mean()
    n = data.groupby('CRIT').size()

    df = pd.DataFrame({'T score': t, 'p values': p, 'n': n}, index=t.index)
    df = df.sort_index()
    print(df.map('{:.4f}'.format))

def query_stat_signif():

    data = pd.read_csv("P5/_log2_knn_summary_file.csv")
    data = data[data['stat_signif'] == 1]
    data = data.sort_values(by="filename", ascending=False)
    print(data['filename'])
    
    
def best_improvement_in_accuracy():
    data = pd.read_csv("P5/_log_knn_summary_file.csv")
    data['acc_diff'] = data['classifier_mean'] - data['majority_class_mean']
    sorted_data = data.sort_values(by="acc_diff", ascending=False)
    print(sorted_data)


def pairwise_ttest_knn(local_dir):
    from scipy.stats import ttest_ind

    data = pd.read_csv(f"{local_dir}_summary_file.csv")

    variables = {}

    for row in data['filename']:
        split_list = row.split('_')
        for s in split_list:
            if '=' in s:
                key, value = s.split('=')
                if key not in variables:
                    variables[key] = set()
                variables[key].add(value)

    names = list(variables['F'])
    ttest_results = pd.DataFrame(0, index=names, columns=names)
    for g in variables['GROUP']:
        for k in variables['K']:
            for t in variables['T']:
                n = len(variables['F'])
                for i in range(0, n):
                    for j in range(0, n):
                        rows = names[i]
                        columns = names[j]
                        
                        string_builder1 = f"/F={rows}_T={t}_GROUP={g}_K={k}"
                        sample1 = pd.read_csv(f"{local_dir}{string_builder1}")

                        string_builder2 = f"/F={columns}_T={t}_GROUP={g}_K={k}"
                        sample2 = pd.read_csv(f"{local_dir}{string_builder2}")

                        ttest = ttest_ind(sample1['Accuracy'], sample2['Accuracy'], equal_var=False)
                        if (ttest.pvalue < 0.05) & (sample1['Accuracy'].mean() > sample2['Accuracy'].mean()):
                            ttest_results.loc[rows, columns] += 1
    
    return ttest_results

def best_feature_subset_table():
    r1=pairwise_ttest_knn(local_dir="P5/_log_knn")
    r2=pairwise_ttest_knn(local_dir="P5/_log2_knn")
    res = r1 + r2
    print(res)


def _log3_analysis(local_dir):
    data = pd.read_csv(f"{local_dir}_summary_file.csv")
    variables = {}

    for row in data['filename']:
        split_list = row.split('_')
        for s in split_list:
            if '=' in s:
                key, value = s.split('=')
                if key not in variables:
                    variables[key] = set()
                variables[key].add(value)
    
    print(data.columns)
    for t in variables['T']:
        for k in variables['K']:
            print(f"T={t}, K={k}")
            subset = data[(data['T'] == t) & (data['K'] == int(k))]
            subset2 = subset[subset['stat_signif']==1]
            print(len(subset2))




def _log3_histogram(local_dir):
    data = pd.read_csv(f"{local_dir}_summary_file.csv")
    import matplotlib.pyplot as plt

    hist_data = data[['GROUP', 'stat_signif']]
    hist_data['GROUP'].astype(int)
    hist_data = hist_data.sort_values(by="GROUP")

    query_init = pd.Series(0, index= hist_data['GROUP'].unique())
    query = hist_data[hist_data['stat_signif'] == 1].groupby(by="GROUP").size()
    query_init.update(query)
    print(query_init)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(query_init.index, query_init.values, width=4, align='center', color='skyblue', edgecolor='black')
    plt.xlabel("GROUP", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.title("# of Significantly Better Classifiers", fontsize=18)
    plt.xticks(query_init.index, fontsize=14)
    plt.yticks([0,5,10,15], fontsize=14)

    plt.axhline(y=15, color='green', linestyle='-', linewidth=1.5)
    plt.show()


def _log_TREE_query(local_dir): 
    data = pd.read_csv(f"{local_dir}_summary_file.csv")

    query_init = pd.Series(0, index=data['CRIT'].unique())
    query = data[data['stat_signif'] == 1].groupby(by='CRIT').size()
    query_init.update(query)
    print(query_init)

    query_init = pd.Series(0, index=data['T'].unique())
    query = data[data['stat_signif'] == 1].groupby(by='T').size()
    query_init.update(query)
    print(query_init)


def pairwise_ttest_TREE(local_dir):
    from scipy.stats import ttest_ind

    data = pd.read_csv(f"{local_dir}_summary_file.csv")
    variables = {}

    for row in data['filename']:
        split_list = row.split('_')
        for s in split_list:
            if '=' in s:
                key, value = s.split('=')
                if key not in variables:
                    variables[key] = set()
                variables[key].add(value)
    print(variables)
    names = list(variables['F'])
    names = sorted(names)
    ttest_results = pd.DataFrame(0, index=names, columns=names)
    for g in variables['GROUP']:
        for t in variables['T']:
            for crit in variables['CRIT']:
                n = len(variables['F'])
                for i in range(0, n):
                    for j in range(0, n):
                        rows = names[i]
                        columns = names[j]
                        
                        string_builder1 = f"/F={rows}_T={t}_GROUP={g}_CRIT={crit}"
                        sample1 = pd.read_csv(f"{local_dir}{string_builder1}")

                        string_builder2 = f"/F={columns}_T={t}_GROUP={g}_CRIT={crit}"
                        sample2 = pd.read_csv(f"{local_dir}{string_builder2}")

                        ttest = ttest_ind(sample1['Accuracy'], sample2['Accuracy'], equal_var=False)
                        if (ttest.pvalue < 0.05) & (sample1['Accuracy'].mean() > sample2['Accuracy'].mean()):
                            ttest_results.loc[rows, columns] += 1
    print(ttest_results)
    ttest_results.to_csv("P6/_log_TREE_pairwise_ttests.csv", index=True)
    return ttest_results



if __name__ == '__main__':
    print("Nothing Happened!")
