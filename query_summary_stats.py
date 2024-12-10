
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


def pairwise_ttest_knn():
    from scipy.stats import ttest_ind

    local_dir = "./_log/_knn_trial1"
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
    
    order = ['vote-count', 'all', 'textblob', 'chi-test', 'textblob-extra']
    ttest_results = ttest_results.reindex(order, axis=0).reindex(order, axis=1)


    print(ttest_results)

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


def pairwise_ttest_TREE():
    from scipy.stats import ttest_ind

    local_dir = "_RF_hours"
    data = pd.read_csv(f"./_log/{local_dir}_summary_file.csv")
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
                        
                        string_builder1 = f"F={rows}_T={t}_GROUP={g}_CRIT={crit}"
                        sample1 = pd.read_csv(f"./_log/{local_dir}/{string_builder1}")

                        string_builder2 = f"F={columns}_T={t}_GROUP={g}_CRIT={crit}"
                        sample2 = pd.read_csv(f"./_log/{local_dir}/{string_builder2}")

                        ttest = ttest_ind(sample1['Accuracy'], sample2['Accuracy'], equal_var=False)
                        if (ttest.pvalue < 0.05) & (sample1['Accuracy'].mean() > sample2['Accuracy'].mean()):
                            ttest_results.loc[rows, columns] += 1
    
    order = ['cate1', 'cate2', 'cate3', 'cate-all', 'cate1-recc', 'cate2-recc', 'cate3-recc',
             'cate-all-recc', 'every', 'textblob', 'textblob-extra', 'vote-count', 'continuous-all']
    ttest_results = ttest_results.reindex(order, axis=0).reindex(order, axis=1)

    ttest_results = ttest_results.rename(
        index=dict(zip(ttest_results.index[:9], range(1,10))),
        columns=dict(zip(ttest_results.columns[:9], range(1,10)))
    )

    # ttest_results.columns[:9] = new_labels
    print(ttest_results)
    ttest_results.to_csv("_log/_RF_hours_pairwise_ttests.csv", index=True)
    return ttest_results

def categorized_line_plot_SHAPE():
    '''
    used for pairwise feature subset analysis
    '''
    data = pd.read_csv("./_log/_recommend_9_hopeful_summary_file.csv")
    # data = data[data['T'] == "freq"]


    # Plot
    fig, ax = plt.subplots()
    default_colors = plt.cm.tab10.colors
    medians = {}

    # Rename Feature Subsets
    feature_name_list = ['cate1', 'cate2', 'cate3', 'cate-all', 
                        'cate1-recc', 'cate2-recc', 'cate3-recc', 'cate-all-recc', 'every']
    for idx, feat in enumerate(feature_name_list):
        data.loc[data['F'] == feat, 'F'] = idx + 1

    # Plot t_scores
    for i, (category, values) in enumerate(data.groupby('F')):
        medians[category] = values['t_score'].median()
        color = default_colors[i%10]
        x_values = values['t_score']
        # Set different shapes for different values of k
        shape = values['CRIT'] == "random"
        shape = shape.map({True: '*', False: 'o'})
        y_values = [str(category)] * len(x_values)  # Repeat category name for y-axis

        for x, y, marker in zip(x_values, y_values, shape):
            ax.scatter(x, y, label=category, marker=marker, color=color)

    # Cluster t scores using k means clustering
    # cluster = KBinsDiscretizer(n_bins=len(data['F'].unique()), strategy="kmeans")
    # cluster.fit(data['t_score'].to_numpy().reshape(-1,1))
    # cluster_edges = cluster.bin_edges_[0]
    # # Add vertical lines for cluster edges
    # for edge in cluster_edges[1:-1]:
    #     ax.axvline(x=edge, color='grey', linestyle='--')

    # Add vertical lines for medians-- this paragraph is written by ChatGPT
    categories = [str(cat) for cat in medians.keys()]
    positions = {cat: i for i, cat in enumerate(categories)}
    for category, median_val in medians.items():
        y_pos = positions[str(category)]
        ax.plot(
            [median_val, median_val],  # x-coordinates for vertical line
            [y_pos - 0.4, y_pos + 0.4],  # Slight offset to span within the category
            color='grey',
            linestyle='--'
        )

    # Set axis labels
    ax.set_title("Feature Subset Analysis")
    ax.set_xlabel('t scores')
    ax.set_ylabel('Feature Subsets')

    plt.tight_layout()

    
    # Create custom legend 
    legend_handles = [
        mlines.Line2D([], [], marker='*', color='w', markerfacecolor='black', markersize=10, label='CRIT = random'),
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor='black', markersize=10, label='CRIT = other')
    ]
    ax.legend(handles=legend_handles, loc='lower right')
    
    plt.savefig("./plots/Feature_Subset_Analysis/rec_9_hopeful_F_SHAPE_analysis.png")
    plt.show()


def categorized_line_plot():
    '''
    used for pairwise feature subset analysis
    '''
    local_dir = "_recommend_textblob"
    data = pd.read_csv(f"./_log/{local_dir}_summary_file.csv")
    # data = data[data['T'] == "freq"]
    # data = data[data['CRIT'] == "entropy"]

    # # Set the order for Feature Subsets
    # order = ['cate1', 'cate2', 'cate3', 'cate-all', 'cate1-recc', 'cate2-recc', 'cate3-recc',
    #          'cate-all-recc', 'every', ]
    #         #'textblob', 'textblob-extra', 'vote-count', 'continuous-all']
    # order = order[::-1]
    # data['F'] = pd.Categorical(data['F'], categories=order, ordered=True)
    # data = data.sort_values('F')

    # # Rename Feature Subsets
    # feat_names = ['cate1', 'cate2', 'cate3', 'cate-all', 'cate1-recc', 'cate2-recc', 'cate3-recc',
    #          'cate-all-recc', 'every']
    # data['F'] = data['F'].astype(str)
    # for idx, feat in enumerate(feat_names):
    #     data.loc[data['F'] == feat, 'F'] = idx+1

    # Plot
    fig, ax = plt.subplots()
    default_colors = plt.cm.tab10.colors
    medians = {}
    # Plot t_scores
    for i, (category, values) in enumerate(data.groupby('F')):
        # use string builder to get all 30 classifier accuracies
        filename = values['filename']
        m = []
        for file in filename:
            print(file)
            accuracies = pd.read_csv(f"./_log/{local_dir}/{file}")
            accuracies = accuracies['Accuracy']
            accuracies = np.random.choice(accuracies.to_numpy(), size=5, replace=False)

            m.append(np.mean(accuracies))
            x_values = accuracies

            # medians[category] = values['t_score'].median()
            color = default_colors[i%10]
            # x_values = values['t_score']
            y_values = [str(category)] * len(x_values)  # Repeat category name for y-axis
            ax.scatter(x_values, y_values, color=color)
        medians[category] = np.mean(np.array(m))

        

    # Cluster t scores using k means clustering
    # cluster = KBinsDiscretizer(n_bins=len(data['F'].unique()), strategy="kmeans")
    # cluster.fit(data['t_score'].to_numpy().reshape(-1,1))
    # cluster_edges = cluster.bin_edges_[0]
    # Add vertical lines for cluster edges
    # for edge in cluster_edges[1:-1]:
    #     ax.axvline(x=edge, color='grey', linestyle='--')

    # Add vertical lines for medians-- this paragraph is written by ChatGPT
    categories = [str(cat) for cat in medians.keys()]
    positions = {cat: i for i, cat in enumerate(categories)}
    for category, median_val in medians.items():
        y_pos = positions[str(category)]
        ax.plot(
            [median_val, median_val],  # x-coordinates for vertical line
            [y_pos - 0.4, y_pos + 0.4],  # Slight offset to span within the category
            color='grey',
            linestyle='--',
            linewidth=2
        )

    # Set axis labels
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Feature Subset')

    plt.tight_layout()

    ax.axvline(x=0.8054451458, color='red', linestyle=':', linewidth=2)
    # Custom Legend
    legend_handles = [
        mlines.Line2D([], [], linestyle='--', color='grey', linewidth=2, label='Median Acc.'),
        mlines.Line2D([], [], linestyle=':', color='red', linewidth=2, label='Majority Class Acc.'),
    ]
    ax.legend(handles=legend_handles, loc='lower left')
    
    plt.savefig(f"./plots/{local_dir}/F_new_acc.png")
    plt.show()

if __name__ == '__main__':
    # categorized_line_plot()
    data = pd.read_csv("./data/data_original/steam_game_reviews.csv")
    print(len(data))