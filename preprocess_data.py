import pandas as pd
import numpy as np


def the_great_sales_merger():
    '''
    Merges my Steam review data with my game sales data.
    Saves the result as a .csv file
    There are 26 matches 
    '''

    # game_name, genre, developer, publisher, overall_player_rating, sales
    steam = pd.read_csv("data\\data_original\\games_description.csv")
    sales = pd.read_csv("data\\data_original\\vgsales.csv") # 26 matches

    # format steam_games
    steam_games = steam[["name", "genres", "developer", "publisher", "overall_player_rating"]]
    steam_games["name"] = steam_games["name"].str.lower()
    print("steam_games len: ", len(steam_games))

    # format game_sales
    sales = sales[['Name','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']]
    sales = sales.rename(columns={'Name':'title'})
    sales["title"] = sales["title"].str.lower()

    # print length
    count_original = len(sales)
    sales = sales.dropna() # drops row count from 64,016 to 1,210
    count_nan = count_original - len(sales)
    sales = sales.drop_duplicates("title", keep="first")
    count_dupes = count_original - count_nan - len(sales)
    games_csv = pd.merge(steam_games, sales, left_on="name", right_on="title", how="inner")
    count_matches = len(games_csv)

    print("Original sales data size: ", count_original)
    print("Number of dropped nan: ", count_nan)
    print("Number of dropped duplicates: ", count_dupes)
    print("Number of matches: ", count_matches)

    games_csv = games_csv.rename(columns={'name':'game_name'})
    games_csv["genres"] = games_csv.pop("genres")
    games_csv.to_csv("data\\data_slim\\games_and_sales.csv", index=False)

################## MAKE REVIEW INFO ##################
def textblob_review_analysis():
    '''
    Tosses my data through TextBlob to generate: number of words, number of sentences, polarity score, 
    and objectivity score. These metrics are stored along with other data from my Steam reviews dataset. 
    This function takes approximatly 15 mintues to run with a dataset of size 992,153.
    '''
    # review_only.csv =     game_name, username, review_text
    # review_info.csv =     game_name, username, textblob_polarity, textblob_objectivity, word_length, sentence_length
    review_dataset = pd.read_csv("data\\data_original\\steam_game_reviews.csv")
    game_dataset = pd.read_csv("data\\data_original\\games_description.csv")
    dataset = pd.merge(review_dataset, game_dataset, how="inner", left_on="game_name", right_on="name")

    review_info = dataset[
        ["game_name", "username","date", 'hours_played','helpful','funny','review',
        'recommendation', 'overall_player_rating', 'publisher', 'developer', 
        'number_of_reviews_from_purchased_people','number_of_english_reviews']]

    review_info = review_info.dropna(subset=['funny', 'helpful'])
    review_info["game_name"] = review_info["game_name"].str.lower()
    review_info['username'] = review_info['username'].str.split('\n').str[0]
    review_info['hours_played'] = review_info['hours_played'].str.replace(',', '', regex=False)
    review_info['helpful'] = review_info['helpful'].str.replace(',', '', regex=False)
    review_info['funny'] = review_info['funny'].str.replace(',', '', regex=False)
    print(review_info.columns)
    print("Size of data: ", len(review_info))

    # review word length
    from textblob import TextBlob
    textblob_data = np.zeros((len(review_info), 4))
    for idx, row in review_info.iterrows():
        text = TextBlob(str(row["review"]))
        info = np.array([len(text.words), len(text.sentences), *text.sentiment]) 
        textblob_data[idx] = info
        if (idx + 1) % 1000 == 0: # TODO: attractive progress bar in the windows terminal
            print(f"Textblob: [{idx+1} / {len(textblob_data)}]")

    review_info[["n_words", 'n_sentences', 'polarity', 'objectivity']] = textblob_data
    review_info.to_csv("data\\data_cache\\textblob_review_analysis.csv") 
    return review_info


def discritize_data(data: pd.DataFrame):
    '''
    discritizes my dataset 3 times:
        polarity -- split unifomly into 3 groups
            negative = [-1, -1/3] 
            neutral = [-1/3, 1/3]
            positive = [1/3, 1]
        objectivity -- split uniformly into 3 groups
            objective = [0, 1/3]
            neutral = [1/3, 2/3]
            subjective = [2/3, 1]
        hours_played -- split into 4 groups
            (1) = pd.qcut() funciton
            (2) = pd.cut() function
    '''
    data = data[data['hours_played'] < 1000]

    polarity_labels = ["negative", "neutral", 'positive']
    data["polarity_disc"] = pd.cut(data["polarity"], bins=3, labels=polarity_labels)

    objectivity_labels = ["objective", "neutral", "subjective"]
    data["objectivity_disc"] = pd.cut(data["objectivity"], bins=3, labels=objectivity_labels)

    data["hours_played_disc1"] = pd.qcut(data["hours_played"], q=8, labels=[0,1,2,3,4,5,6,7])
    data["hours_played_disc2"] = pd.cut(data["hours_played"], bins=8, labels=[0,1,2,3,4,5,6,7])

    return data


def knn_ready_data():
    '''
    features = data[['helpful','funny','polarity','subjectivity','n_words','n_sentences']]
        -- hours_played < 1000
        -- drop NaN helpful, funny
        -- normalize
        -- no discretization of polarity and subjectivity
    targets = 'hours_played'
        -- seperate pandas series
        -- pd.cut() and pd.qcut() then to numpy
    '''

    data = pd.read_csv("data/train/data_about_review.csv")

    data = data.dropna(subset=['helpful', 'funny'])
    data = data[data['hours_played'] <= 1000]
    features = data[['helpful','funny','polarity','subjectivity','n_words','n_sentences']]
    features = features.to_numpy()

    # normalize all continuous features using z-score normaliztion
    for col in range(features.shape[1]):
        mean = features[col].mean()
        std = features[col].std()
        
        # Handle case where standard deviation is 0
        if std == 0:
            features[col] = 0
        else:
            features[col] = (features[col] - mean) / std

    np.save("data/train/knn_ready_features.npy", features)

    targets = data[['hours_played']]

    targets.to_csv("data/train/knn_ready_target.csv", index=False)


def tree_ready_data():
    '''
    save dataset as np.matrix with 

    features = data[['helpful','funny','polarity','subjectivity','n_words','n_sentences']]
        -- hours_played < 1000
        -- drop NaN helpful, funny
        -- normalize
        -- no discretization of polarity and subjectivity
    targets = 'hours_played'
        -- seperate pandas series
        -- pd.cut() and pd.qcut() then to numpy
    '''

    data = pd.read_csv("data/train/data_about_review.csv")

    data = data.dropna(subset=['game_name', 'publisher', 'developer', 'overall_player_rating'])
    data = data.dropna(subset=['helpful', 'funny'])
    data = data[data['hours_played'] <= 1000]
    features = data[['helpful','funny','polarity','subjectivity','n_words','n_sentences',
                     'game_name', 'publisher', 'developer', 'overall_player_rating', 'recommendation']]

    features.to_csv("data/train/tree_ready_features.csv")

def train_test_split(data: pd.DataFrame):
    '''
    splits data into test and train using sklearn's stratified k fold with hours_played_disc1 as my targets
    Uses set random seed of 42
    90% train
    10% test
    '''

    from sklearn.model_selection import StratifiedKFold
    random_state = 42
    skf = StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)

    targets = data["hours_played_disc1"]
    features = data.drop(columns=["hours_played_disc1", "hours_played_disc2", "hours_played", "polarity", "objectivity"])
    
    for train_indices, test_indices in skf.split(features, targets):
        return data.iloc[train_indices], data.iloc[test_indices]


def main():
    '''
    DATA:
        -- Kaggle. Steam Games, Reviews, and Rankings. https://www.kaggle.com/datasets/mohamedtarek01234/steam-games-reviews-and-rankings?select=steam_game_reviews.csv
        -- Kaggle.  Video Game Sales. https://www.kaggle.com/datasets/gregorut/videogamesales 
    ''' 

    # CAUTION: this will take 20+ minutes to complete on my 1,000,000 size dataset
    # textblob_review_analysis()

    data = pd.read_csv("data\\data_cache\\textblob_review_analysis.csv")
    data = discritize_data(data)

    train, test = train_test_split(data)

    train_review = train['review']
    train = train.drop(columns=["review"])
    test_review = test['review']
    test = test.drop(columns=['review'])

    train.to_csv("data\\train\\data_about_review.csv")
    train_review.to_csv("data\\train\\review.csv")
    test.to_csv("data\\test\\data_about_review.csv")
    test_review.to_csv("data\\test\\review.csv")

