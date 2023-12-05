# WEEK 10 HW - JaMor Hairston

import numpy as np
import pandas as pd
from bertopic import BERTopic
import re

# 1. Use the [COVID-19 Rumor Dataset](https://github.com/MickeysClubhouse/COVID-19-rumor-dataset/tree/master/Data/twitter) 
    # to perform Dynamic Topic Modeling with `nr_bins = 25`.

# Load and prepare the dataset
trump = pd.read_csv('https://github.com/MickeysClubhouse/COVID-19-rumor-dataset/tree/master/Data/twitter')
#trump = pd.read_csv('1002962201143611433.csv')

# Filter
trump.text = trump.apply(lambda row: re.sub(r"http\S+", "", row.text).lower(), 1)
trump.text = trump.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.text.split())), 1)
trump.text = trump.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.text).split()), 1)
trump = trump.loc[(trump.isRetweet == "f") & (trump.text != ""), :]
timestamps = trump.date.to_list()
tweets = trump.text.to_list()

topic_model = BERTopic(min_topic_size=35, verbose=True)
topics, probs = topic_model.fit_transform(tweets)

topics_over_time = topic_model.topics_over_time(tweets, timestamps, nr_bins=25)


# 2. Visualize the trends in any 5 topics over time and report your observations.

topic_model.visualize_topics_over_time(topics_over_time, topics=[9, 10, 72, 83, 87, 91])
