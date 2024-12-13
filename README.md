
## Data

A web scraping of Steam reviews by Mohamad Tarek yielded this dataset. The data describes review messages left by
Steam community members across 290 games. I use a Natural Language Processing library, TextBlob, to produce
sentiment scores and length measurements. I use these features to predict if a user recommends a game (binary) and
the number of hours that the user played that game. 

Data can be found at:
https://www.kaggle.com/datasets/mohamedtarek01234/steam-games-reviews-and-rankings 

    
## Discussion

Predicting hours played is problematic. It is a continuous variable and my Data Mining class focused on predictions
for categoricals. In my first step I discretized hours played into categorical bins. I tried 4 approaches: K-means
clusters, equal width bins, equal frequency bins, and equal width bins after taking the logarithm of hours played. 
I discovered equal frequency to be conclusively the best disretization method (see `./plots/Random_Forest_hparam_sweep/Target_discretization_analysis.png`). The next issue arises when selecting the number of bins to disretize into. `./plots/Random_Forest_hparam_sweep/GROUP_5_100_percentage_Validation_bs=3.png` and `./plots/Random_Forest_hparam_sweep/GROUP_1_20_percentage_Validation_bs=3.png` are two analyses I performed to motivate my bin size choice. I chose a bin size of 8 because it is relatively small and it's a local maxima. However, a bin size of 80 actually performs better on my metric. 

My next choice for analysis was to group sets of features together to test if I could find a subset of features which predicted hours played or recommendation better than using all of my features. Slides 12-17 of my presentation (`./slides/Slide_Deque.pptx`) are an organization of my exploration of feature subsets. Slide 14 shows where I found a subset which outperformed using every feature!

Finally, I wrote a script which trains models for 10 different Scikit-Learn classifiers. My scipt can be found at `run_all_classifiers.py` and my plots are on slides 18, 20, and 21 of my presentation. 

In the end, my best predictive models were a Random Forest to predict recommendation and a Neural Network to predict hours played. I'm disapointed that these were the two models I selected. Training a Random Forest or Neural Network are likely the first models a Data Scientist would apply to this data, however, nothing that I learning in my validation set motivated me to select any different model. Random Forest and Neural Network classifiers are state of the art for a reason. Additionally, I could expand on my search of feature subsets by utilizing Associative Rule Mining techniques. Associative Rule Mining essentially searches for relationships between features or groups of features. My classifying Neural Network could have benefited from a more exhaustive hyperparameter search. I am new to the theory of Neural Networks and that was not the focus of this class, so I chose to keep it simple. 


## File structure

- **`categorized_line_plot.py`**
    The best plot for comparing differences between classifiers. I can grouby any attribute then plot
    the distribution of t scores. I have a dashed line to represent the median, which is an unbiased
    estimator. I also have the option for representing certain features with shapes. 
- **`classifier.py`**
    My implementation of the K Nearest Neighbors and Naive Bayes algorithms.
- **`cluster_visualization.py`**
    A plot for visualizing 1D cluster groups. Each cluster border is represented with an edge. Higher
    sloped lines show tigher cluster groups. Each line uses a different linestyle to appeal to color
    blinded individuals.
- **`decision_tree.py`**
    My implementation of the Decision Tree classification algorithm.
- **`dependencies.py`**
    Prints a list of dependencies for a folder of python files. Used in my README.md
- **`eval_on_test_set.py`**
    Evaluation on the test set across my 3 different groups of experiments. I ran KNN to predict hours played,
    Random Forest to predict hours played, and Random Forest to predict recommendation.
- **`log_analyze.py`**
    Iterates through my _log files and calculates summary statistics. Most importantly, this script
    performs the t-test to test if my classifier is significantly better than the majority class
    baseline. Saves a summary file averaged over 30 runs. Allows for t-test analysis. 
- **`log_query.py`**
    Every analysis of my _log files is here. Each method creates a table or histogram that I use
    in my slides. 
- **`log_rename.py`**
    Iterates through a folder and renames files. 
- **`preprocess_data.py`**
    Collection of functions which are part of the preprocessing step. Runs my review messages through 
    TextBlob sentiment analysis to generate new features. Discretizes hours played and prepares my data
    for my KNN and Random Forest algorithms. Finally, the data is split into train and test. 
- **`random_forest.py`**
    My implementation of a Random Forest. I use an ensembling of Decision Trees to achieve a Random Forest. 
- **`run_all_classifiers.py`**
    Fits my dataset to several Sklearn classifiers. This is so useful in comparing performance across 
    many classifying algorithms. Models can be fit quickly using bootstrapping to create a subset of data. 
    This program outputs the data in a compatible format with my `categorized_line_plot.py` which allows 
    for easy visualization of results.     
- **`run_knn.py`** 
    This file runs my K Nearest Neighbors classifier on my data using
    several feature subsets, target discretization methods, and hyperparameter choices. 
    The script will generate _log files which stores data about classifier performance compared
    with the majority classifier.
- **`run_random_forest.py`**
    Same functionality of run_knn.py but uses sklearn's random forest to classify. 
    Also pulls sklearn metrics, which I will be using going forward. Saved in a standard format
    to my _log folder.


## Dependencies
    matplotlib==3.9.0
    numpy==1.26.0
    pandas==2.2.2
    scikit-learn==1.5.2
    scipy==1.14.1
    tyro==0.8.6

