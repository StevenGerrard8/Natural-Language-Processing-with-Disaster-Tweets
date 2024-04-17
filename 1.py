import numpy as np
import pandas as pd
from glove import Corpus, Glove

corpus = Corpus()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


def lower_case(x):
    return x.lower()


train['text'] = train['text'].apply(lower_case)
test['text'] = test['text'].apply(lower_case)
train = train.drop(['id', 'keyword', 'location'], axis=1)
test = test.drop(['id', 'keyword', 'location'], axis=1)
X_train = train['text'].values.reshape(-1, 1)  # Reshape to a column vector
X_test = test['text'].values.reshape(-1, 1)

# Combine train and test data
total_tweets = np.concatenate((X_train, X_test))
all_words = []

# Iterate over each row of total_tweets
for tweet in total_tweets:
    # Split the tweet into words

    words = str(tweet).split()  # Assuming each row contains a single text value
    # Extend the list of all words with the words from the current tweet
    all_words.append(words)
# Initialize Corpus and fit to data
corpus = Corpus()
corpus.fit(all_words, window=10)

# Train GloVe model
glove = Glove(no_components=31, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

# Save trained model
glove.save('your_trained_glove_model')