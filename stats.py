import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
import string
import pandas as pd
import numpy as np
from data_etl import Data
from keras.models import Sequential
from keras import layers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import warnings


warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


'''
Initial organization
'''
def organize(movie_data):
    data_manager = movie_data
    merged, wikidata, omdb, rt = data_manager.get_all_data()
    merged['publication_date'] = pd.to_datetime(merged['publication_date'])

    # convert dates to year
    merged['publication_date'] = merged['publication_date'].dt.year
    merged['publication_date'] = merged['publication_date'].values.astype(int)
    merged = merged.dropna(subset=['audience_average'])

    # keep certain columns
    merged = merged[
        ['imdb_id', 'enwiki_title', 'omdb_genres', 'publication_date', 'wikidata_id', 'omdb_plot', 'audience_average',
         'critic_average']]
    merged = merged.dropna()
    return merged


'''
Create a variety of statistical plots
'''
def create_stats(movie_data):
    merged = movie_data

    # -------- see the histograms of ratings for genres -----------
    genres = merged[["imdb_id", 'enwiki_title', 'publication_date', 'omdb_genres', 'audience_average']]
    genres = genres.explode('omdb_genres')
    # keep genres with more than 50 occurences
    genres = genres.dropna(subset=['omdb_genres'])
    unpopular_genres = ["N/A", "Short", "News", "Film-Noir"]
    genres = genres[~genres['omdb_genres'].isin(unpopular_genres)]
    genres = genres.drop('enwiki_title', axis=1)

    # amount of data per genre
    count = genres['omdb_genres'].value_counts()
    #print(count)
    plt.figure(0)
    count.plot.bar()
    plt.tight_layout()
    plt.savefig("genre-counts.png")


    # an example plot
    plt.figure(1)
    groups_obj = genres.groupby("omdb_genres")
    group_names = groups_obj.groups.keys()
    plt.hist(groups_obj.get_group("Action").audience_average)
    plt.title("Action Hist")
    plt.savefig("Action.png")

    # ------------ see how genre ratings have changed over time since 1995  to 2018 ------------------
    plt.figure(2)
    fig, ax = plt.subplots(figsize=(15,7))
    popular_genres = ["Music", "Documentary", "Adventure", "Comedy", "Drama"]
    popular_genres_df = genres[genres['omdb_genres'].isin(popular_genres)]
    popular_genres_df = popular_genres_df[popular_genres_df["publication_date"] >= 1995]
    popular_genres_df = popular_genres_df[popular_genres_df["publication_date"] <= 2017]
    popular_genres_means = popular_genres_df.groupby(["omdb_genres"]).aggregate({'audience_average':['mean']})
    popular_genres_df = popular_genres_df.groupby(["publication_date", "omdb_genres"]).aggregate({'audience_average':['mean']})
    popular_genres_df.unstack().plot(ax=ax)
    plt.legend(popular_genres)
    plt.savefig("genres_over_time.png")


'''
Organizes the data
'''
def organize_for_learning(movie_data):
    # setup
    merged = movie_data
    data = merged[['omdb_plot', 'omdb_genres']]
    data = data.explode('omdb_genres')
    # keep genres with more than 50 occurences
    data = data.dropna(subset=['omdb_genres'])
    unpopular_genres = ["N/A", "Short", "News", "Film-Noir"]
    data = data[~data['omdb_genres'].isin(unpopular_genres)]

    # clean up text from plots: use all lowercase, remove puncuation
    data['omdb_plot'] = data['omdb_plot'].apply(lambda x: x.lower())
    data['omdb_plot'] = data['omdb_plot'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    # create an equal amount of data per genre for top 2 genres. 1=drama, 0=comedy
    drama = data[data['omdb_genres'] == "Drama"]
    drama['omdb_genres'] = 1
    comedy = data[data['omdb_genres'] == "Comedy"]
    comedy['omdb_genres'] = 0

    drama = drama.head(2100)
    comedy = comedy.head(2100)
    data_joined = pd.concat([drama, comedy])

    plot_summaries = data_joined['omdb_plot']
    vals = data_joined['omdb_genres']
    return plot_summaries.values, vals.values


'''
Transform plot summaries using tf-idf
'''
def vectorize(X_tr, X_te):
    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit(X_tr)

    X_train = vectorizer.transform(X_tr)
    X_test = vectorizer.transform(X_te)

    tfidf_transformer = TfidfTransformer()

    X_train_tfidf = tfidf_transformer.fit_transform(X_train)
    X_test_tfidf = tfidf_transformer.fit_transform(X_test)

    return X_train_tfidf, X_test_tfidf


'''
A function that tests a variety of machine learning models
'''
def machine_learning(plot_summaries, values):
    print("================================================\n"
          "**********   Models are training...  ***********\n"
          "================================================\n")

    plots = plot_summaries
    y = values

    # split into training and testing sets
    Plot_train, Plot_test, y_train, y_test = train_test_split(plots, y, test_size=0.25, random_state=1000)

    # vectorize text
    X_train, X_test = vectorize(Plot_train, Plot_test)

    # models to test out
    MNB = ("Multinomial Naive Bayes", MultinomialNB())
    LR = ("Logistic Regression", LogisticRegression())
    L_SVC = ("SVC", SVC(C=1.0, kernel='linear', degree=2, gamma='auto'))
    KN = ("k-nearest neighbours classifier", KNeighborsClassifier(n_neighbors=100))

    models = [MNB, LR, L_SVC, KN]

    for model in models:
        model[1].fit(X_train, y_train)
        y_predicted = model[1].predict(X_test)
        score = model[1].score(X_test, y_test)

        cm = confusion_matrix(y_test, y_predicted)
        fig = plt.figure()
        plt.matshow(cm)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.title(model[0])
        plt.savefig(model[0] + '.png')

        print("Accuracy for " + model[0] + " using TF-IDF: ", score)


'''
A function that tests neural networks using keras
'''
def neural_networks(plot_summaries, values):
    plots = plot_summaries
    y = values

    # split into training and testing sets
    Plot_train, Plot_test, y_train, y_test = train_test_split(plots, y, test_size=0.25, random_state=1000)

    # vectorize text
    X_train, X_test = vectorize(Plot_train, Plot_test)

    # use keras for simple neural network
    input_dim = X_train.shape[1]  # number of features

    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # model.summary()

    print("=============================================================\n"
          "*************   Neural networks are training...  ************\n"
          "=============================================================\n")

    history = model.fit(X_train, y_train,
                        epochs=40,
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=32)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))


def main():
    movie_data = organize(Data())  # initialize data set
    create_stats(movie_data)
    plot_summaries, values = organize_for_learning(movie_data)
    machine_learning(plot_summaries, values)  # train and test machine learning models
    neural_networks(plot_summaries, values)  # train and test neural networks


if __name__ == '__main__':
    main()
