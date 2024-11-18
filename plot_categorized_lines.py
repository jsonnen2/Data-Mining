import pandas as pd
import matplotlib.pyplot as plt
import mlines

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

    
if __name__ == '__main__':
    '''
    used for pairwise feature subset analysis
    '''
    data = pd.read_csv("./_log/_knn_trial2_summary_file.csv")
    data = data[data['T'] == "freq"]

    # Plot
    fig, ax = plt.subplots()
    default_colors = plt.cm.tab10.colors

    # Rename Feature Subsets
    # feature_name_list = ['cate1', 'cate2', 'cate3', 'cate-all', 
    #                     'cate1-recc', 'cate2-recc', 'cate3-recc', 'cate-all-recc', 'every']
    # for idx, feat in enumerate(feature_name_list):
    #     data.loc[data['F'] == feat, 'F'] = idx + 1

    medians = {}
    # Plot t_scores
    for i, (category, values) in enumerate(data.groupby('GROUP')):
        medians[category] = values['t_score'].median()
        color = default_colors[i%10]
        x_values = values['t_score']
        y_values = [str(category)] * len(x_values)  # Repeat category name for y-axis
        ax.scatter(x_values, y_values, color=color)

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
            linestyle='--'
        )

    ax.axvline(x=2.04, color='red', linestyle='--', linewidth=2)

    # Set axis labels
    ax.set_title("GROUP size analysis")
    ax.set_xlabel('t scores')
    ax.set_ylabel('Bin size')

    plt.tight_layout()

    legend_handles = [
        mlines.Line2D([], [], linestyle='--', color='red', linewidth=2, label='signifigance level'),
    ]
    ax.legend(handles=legend_handles, loc='upper right')
    
    plt.savefig("./plots/KNN/trial2_GROUP_analysis.png")
    plt.show()