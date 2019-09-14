import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_etl import Data
from scipy import stats

data_manager = Data()
merged, wikidata, omdb, rt = data_manager.get_all_data()

# success criteria correlation analysis
criteria_cols = ["audience_average", "audience_percent", "critic_average", "critic_percent"]
criteria = merged[criteria_cols].dropna()
corr = criteria.corr()

corr_matrix = plt.figure(figsize=(20, 15))
plt.matshow(corr, fignum=corr_matrix.number)
plt.title("Success Criteria Correlation Matrix", fontsize=28, y=1.10);
plt.xticks(range(len(criteria_cols)), criteria_cols, fontsize=20)
plt.yticks(range(len(criteria_cols)), criteria_cols, fontsize=20)
cb = plt.colorbar()
plt.savefig("correlation-matrix")

# genre analysis
genres = merged.dropna(subset=["audience_average"])
genres = merged[["imdb_id", "omdb_genres", "audience_average"]].dropna()
genres = genres.explode("omdb_genres")
genres = genres.dropna(subset=["omdb_genres"])

# genres to keep
counts = genres.groupby("omdb_genres").size().to_frame("size").reset_index()
keep = counts[counts["size"] >= 50].reset_index()
keep = keep["omdb_genres"].values

# get ratings
genres = genres[genres["omdb_genres"].isin(set(keep))]
genres = genres.sort_values(by=["omdb_genres"])
ratings_df = genres.pivot(columns="omdb_genres", values="audience_average")
genre_ratings = [ratings_df[genre].dropna() for genre in keep]
means = [genre.mean() for genre in genre_ratings]
std_devs = [genre.std() for genre in genre_ratings]

# ANOVA
print("ANOVA for ratings of all genres: ")
print(stats.f_oneway(*genre_ratings), end="\n\n")

# ANOVA for the top 6 genres in terms of counts
print("ANOVA for ratings of Drama, Comedy, Action, Adventure, Crime, and Romance: ")
highest_counts = ["Drama", "Comedy", "Action", "Adventure", "Crime", "Romance"]
highest_counts_ratings = [ratings_df[genre].dropna() for genre in highest_counts]
print(stats.f_oneway(*highest_counts_ratings), end="\n\n")

# Bar graph
plt.figure(figsize=(25, 20))
plt.bar(keep, means, yerr=std_devs)
plt.gca().set_ylim([0, 10])
plt.xticks(range(len(means)), keep, rotation=50, fontsize=20)
plt.yticks(range(11), [str(i) for i in range(0, 11)], fontsize=20)

plt.title("Average Audience Ratings by Genre", fontsize=28, y=1.10);
plt.xlabel("Movie Genre", fontsize=25)
plt.ylabel("Average Audience Rating (out of 10)", fontsize=25)
plt.savefig("genre-ratings", fontsize=25)