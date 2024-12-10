
import pandas as pd

data = pd.read_csv("./data/train/data_about_review.csv")

col_names = ['helpful','funny','polarity','subjectivity','n_words','n_sentences','hours_played',
            'game_name', 'publisher', 'developer', 'overall_player_rating']

data = data[col_names]

data.to_csv("data/train/predict_recommendation/features.csv", index=False)