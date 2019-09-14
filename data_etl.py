import numpy as np
import pandas as pd

class Data:
  def __init__(self):
    self.data = self.read_data()

  def transform_wikidata(self, wikidata, genres):
    genre_dict = genres["genre_label"].to_dict()

    def map_genre(genre_list):
      return [genre_dict.get(g, g) for g in genre_list]
    map_genre = np.vectorize(map_genre, otypes=[np.object])

    wikidata["genre"] = map_genre(wikidata["genre"])
    return wikidata

  def clean_rt(self, rt):
    # bring audience_average and critic_average to same scale
    rt["audience_average"] = rt["audience_average"] * 2
    return rt

  def read_data(self):
    # read in the data
    wikidata = pd.read_json("./data/wikidata-movies.json.gz", orient="record", lines=True, convert_dates=[13])
    omdb = pd.read_json("./data/omdb-data.json.gz", orient="record", lines=True)
    rt = pd.read_json("./data/rotten-tomatoes.json.gz", orient="record", lines=True)
    genres = pd.read_json("./data/genres.json.gz", orient="record", lines=True).set_index("wikidata_id")
    label_map = pd.read_json("./data/label-map.json.gz", orient="record", lines=True).set_index("wikidata_id")

    # wikidata columns to keep
    wikidata_cols = ["imdb_id", "enwiki_title", "director", "cast_member", "genre", "main_subject", 
      "country_of_origin", "original_language", "nbox", "ncost", "profit", "made_profit",
      "publication_date", "wikidata_id"]
    omdb_cols = ["imdb_id", "omdb_genres", "omdb_plot"]
    rt_cols = ["imdb_id", "audience_average", "audience_percent", "audience_ratings",
      "critic_average", "critic_percent"]

    # clean and transform the DataFrames
    wikidata = wikidata[wikidata_cols]
    omdb = omdb[omdb_cols]
    rt = rt[rt_cols]

    wikidata = self.transform_wikidata(wikidata, genres)
    rt = self.clean_rt(rt)
    merged = (
      wikidata.merge(omdb, on="imdb_id")
      .merge(rt, on="imdb_id")
      .merge(label_map, on="wikidata_id")
    )

    data = {
      "merged": merged, 
      "wikidata": wikidata,
      "omdb": omdb,
      "rt": rt,
    }
    return data

  # get all DataFrames
  def get_all_data(self):
    return self.data["merged"], self.data["wikidata"], self.data["omdb"], self.data["rt"]

  # explode specified col
  def get_explode_df(self, col, select_cols=[]):
    copy = self.data["merged"].copy()
    explode = copy[["imdb_id", col]].set_index("imdb_id")
    del copy[col]

    explode = (
      explode[col]
      .apply(pd.Series)
      .stack()
      .reset_index(level=1, drop=True)
      .to_frame(col)
    )

    copy = copy.merge(explode, on="imdb_id")
    return copy[select_cols] if select_cols else copy