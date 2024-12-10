
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os


if __name__ == '__main__':
# def TEXTBLOB_features():

    # local variables
    eval_set = 'Validation'
    local_dir = "./_log/GROUP_hours_1_20"
    save_dir = f"./plots/_RF_hours/GROUP_1_20_percentage_{eval_set}_bs=3.png"

    # Plot
    fig, ax = plt.subplots()
    default_colors = plt.cm.tab10.colors[:7] + plt.cm.tab10.colors[8:]
    medians = {}

    for i, filename in enumerate(os.listdir(local_dir)):

        file_path = os.path.join(local_dir, filename)
        data = pd.read_csv(file_path)

        x_values = data['Accuracy'] / data['Majority Class Accuracy']
        category = int(filename.split('G=')[1].split('_F')[0])
        y_values = [category] * len(x_values)

        color = default_colors[i%9]
        ax.scatter(x_values, y_values, color=color)
        medians[category] = np.median(x_values)

    # Add vertical lines for medians-- this paragraph is written by ChatGPT
    # categories = [str(cat) for cat in medians.keys()]
    # positions = {cat: i for i, cat in enumerate(categories)}
    # for category, median_val in medians.items():
    #     y_pos = positions[str(category)]
    #     ax.plot(
    #         [median_val, median_val],  # x-coordinates for vertical line
    #         [y_pos - 0.4, y_pos + 0.4],  # Slight offset to span within the category
    #         color='grey',
    #         linestyle='--',
    #         linewidth=2
    #     )
    for ypos, median in medians.items():
        # ax.plot([median, median], [ypos - 0.5, ypos + 0.5], color='grey', linestyle='--', linewidth=2)
        # plot triangle
        x = [median, median-.007, median+.007]
        y = [ypos, ypos-0.5, ypos-0.5]
        ax.fill(x, y, color="yellow", edgecolor="black", linewidth=2)

    # Set axis labels
    # ax.set_xlabel(f'Classifier - Majority')
    ax.set_xlabel(r'Random Forest Accuracy $\div$ Majority Accuracy')
    ax.set_yticks(np.arange(2, 21))
    # ax.set_yticks(np.arange(5, 105, 5))

    plt.tight_layout()
    plt.savefig(save_dir)
    plt.show()


# if __name__ == '__main__':
def categorized_line_plot():
    '''
    used for pairwise feature subset analysis
    '''
    ds_type = 'CATEGORICAL'
    dataset = 'recommend'
    eval_set = 'Test'
    data = pd.read_csv(f"./all_classifiers/TEXTBLOB_recommend_bs=3.csv", index_col=0)
    # data = pd.read_csv(f"./all_classifiers/{ds_type}_{dataset}_3.csv", index_col=0)
    # data = data.drop(index=['QDA', 'Naive Bayes'])
    # data = data.drop(index=['Random Forest', 'Decision Tree'])
    # data_rf = pd.read_csv(f"./all_classifiers/TEXTBLOB_tree_trials_bs=3.csv", index_col=0)
    # data_rbf = pd.read_csv(f"./all_classifiers/RBF_{dataset}_3.csv", index_col=0)
    # data = pd.concat([data, data_rf, ], axis=0)
    print(data.index)
    data = data.iloc[[0,1,4,5,6,7,2,3]]
    
    val_data = data.map(lambda x: float(x.strip("()").split(",")[0]))
    test_data = data.map(lambda x: float(x.strip("()").split(",")[1]))
    
    if eval_set == 'Validation':
        used_data = val_data
    elif eval_set == 'Test':
        used_data = test_data
    else:
        raise NameError("Pick different eval_set")

    # Plot
    fig, ax = plt.subplots()
    default_colors = plt.cm.tab10.colors[:7] + plt.cm.tab10.colors[8:]
    medians = {}
    # Plot t_scores
    for i, (category, acc) in enumerate(used_data.iterrows()):
        x_values = acc
        y_values = [str(category)] * len(x_values)
        medians[category] = acc.median()
        color = default_colors[i%9]
        ax.scatter(x_values, y_values, color=color)

        
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
    ax.set_xlabel(f'{eval_set} Accuracy')

    plt.tight_layout()

    # Custom Legend
    # legend_handles = [
    #     mlines.Line2D([], [], linestyle='--', color='grey', linewidth=2, label='Median Acc.'),
    #     mlines.Line2D([], [], linestyle=':', color='red', linewidth=2, label='Majority Class Acc.'),
    # ]
    # ax.legend(handles=legend_handles, loc='lower left')
    
    plt.savefig(f"./plots/_recommend_textblob/ALL_classifiers_RECOMMEND_{eval_set}_bs=3.png")
    # plt.savefig(f"./plots/ALL_{ds_type}/SUPERSET_{dataset}_{eval_set}_ALL.png")
    plt.show()