"""
This module contains utility classes and functions used across various notebooks in the project.

It includes helper functions and classes that complement the notebooks for the Data Science Method.

This module is part of a larger data analysis project and is used in various Jupyter notebooks.
"""
import concurrent.futures as cf
import itertools as it
import re
from typing import Optional, Callable
from pathlib import Path
from urllib.request import urlopen
from fiona.io import ZipMemoryFile
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import geopandas as gpd
import geoviews as gv
from bokeh.models import NumeralTickFormatter
from thefuzz import fuzz
import hvplot
import hvplot.pandas
import holoviews as hv
from holoviews import streams
import colorcet as cc
import cartopy.crs as ccrs
import panel as pn
import panel.widgets as pnw
import seaborn as sns
from wordcloud import WordCloud
from PIL import ImageDraw, Image
from joblib import Memory, Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
from sklearn import metrics
import unicodedata
from pmdarima import auto_arima
from tqdm import tqdm
import umap

from translate_app import translate_list_to_dict

pn.extension()
hv.extension("bokeh")
gv.extension("bokeh")
hvplot.extension("bokeh")


class InfoMixin:
  """
    Mixin to add a method for limited column info display to a DataFrame.
    """

  def limit_info(self, max_cols=8):
    """
        Display info for up to `max_cols` randomly chosen columns of the DataFrame.

        Parameters
        ----------
        max_cols : int, optional
            Maximum number of columns to display info for. Default is 8.
        """
    if len(self.columns) > max_cols:
      print(f"\nTotal number of columns: {len(self.columns)}")
      self.info(max_cols=max_cols)
      print(f"\nOnly showing info for {max_cols} columns, chosen at random.")
      random_columns = np.random.choice(self.columns,
                                        size=max_cols,
                                        replace=False)
      self[random_columns].info()
    else:
      self.info()


class InfoDataFrame(InfoMixin, pd.DataFrame):
  """
    DataFrame subclass with additional `limit_info` method.
    """


# Set the cache directory
cache_dir = "./zurich_cache_directory"
memory = Memory(cache_dir, verbose=0)

# Opts for polygon elements
poly_opts = dict(
    width=500,
    height=500,
    color_index=None,
    xaxis=None,
    yaxis=None,
    backend_opts={"toolbar.autohide": True},
)

NEIGHBORHOOD_GDF_PATH = "../data/zurich_neighborhoods.geojson"
PROCESSED_DOG_DATA_PATH = "../data/processed_dog_data.csv"
PROCESSED_POP_DATA_PATH = "../data/processed_pop_data.csv"
PROCESSED_INCOME_DATA_PATH = "../data/processed_income_data.csv"

# Create a player widget
yearly_player = pnw.Player(
    name="Yearly Player",
    start=2015,
    end=2020,
    value=2020,
    step=1,
    loop_policy="loop",
    interval=3000,
)
# Create a slider for the roster
roster_slider = pnw.IntSlider(value=2020, start=2015, end=2022)

roster_button = pnw.RadioButtonGroup(
    value=2018,
    options=list(range(2015, 2021)),
    button_style="outline",
    button_type="default",
)

quantile_income_button = pnw.RadioButtonGroup(
    value="median_income",
    options=["lower_q_income", "median_income", "upper_q_income"],
    button_style="outline",
)


def query_for_time_period(df,
                          start_year=2015,
                          end_year=2023,
                          year_col="roster"):
  """Returns a DataFrame with the data for the specified time period,
    querying the 'roster' column (start <= x < end)."""
  try:
    return df.query(f"{start_year} <= {year_col} < {end_year}")
  except NameError as e:
    print(f"Error: {e}")

    return df.query(f"{start_year} <= roster < {end_year}")


def remove_accents(input_str):
  """Function to remove accents from a string"""
  nfkd_form = (unicodedata.normalize("NFKD",
                                     input_str).encode("ASCII",
                                                       "ignore").decode())
  return nfkd_form


def convert_to_snake_case(item):
  """Function to convert a string to snake case"""
  # Add _ before uppercase in camelCase
  s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", item)
  # Add _ before uppercase following lowercase or digit
  s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
  # Add _ between letter and digit
  s3 = re.sub(r"([a-zA-Z])([0-9])", r"\1_\2", s2)
  s4 = re.sub(r"[-\s]", "_", s3).lower()  # Replace hyphen or space with _
  return s4


def sanitize_df_column_names(df):
  """Function to danitize column names by translating and conveting to snake case"""
  column_list = df.columns.tolist()
  # translate the column names
  translated_dict = translate_list_to_dict(column_list)
  # map the translated column names to the column names
  df.rename(columns=translated_dict, inplace=True)
  # convert the column names to snake case
  df.columns = [convert_to_snake_case(col) for col in df.columns]
  return df


def get_gdf_from_zip_url(zip_url: str) -> dict[str, gpd.GeoDataFrame]:
  """Function to get the geojson data from the zip url.
    In the zip url, the geojson files are in the data folder."""
  gpd_dict = {}

  try:
    with urlopen(zip_url) as u:
      zip_data = u.read()
    with ZipMemoryFile(zip_data) as z:
      geofiles = z.listdir("data")
      for file in geofiles:
        with z.open("data/" + file) as g:
          gpd_dict[Path(file).stem] = gpd.GeoDataFrame.from_features(g,
                                                                     crs=g.crs)
  except Exception as e:
    raise Exception(f"Error reading geojson data: {e}")

  return gpd_dict


def rename_keys(d, prefix="zurich_gdf_"):
  """Rename the keys of a dictionary with a prefix."""
  return {f"{prefix}{i}": v for i, (k, v) in enumerate(d.items())}


def get_zurich_description(zurich_description_url: str) -> pd.DataFrame:
  """Function to get the description of the districts of Zurich from the website."""
  # Define regex patterns
  pattern_1 = re.compile(r"s-[1-9]|s-1[0-2]")
  pattern_2 = re.compile(r"([\d]+)")
  # get the html content of the website
  with urlopen(zurich_description_url) as u:
    zurich_html_content = u.read()

  zurich_soup = BeautifulSoup(zurich_html_content, "lxml")

  elements = zurich_soup.find_all(id=pattern_1)

  # create a dataframe with the information of the districts
  districts = {
      element.find("h2").text: element.find("p").text
      for element in elements
  }
  districts_df = pd.DataFrame.from_dict(districts,
                                        orient="index",
                                        columns=["desc"])

  # make the index into a column and split it into district number and district name
  districts_df = districts_df.reset_index()
  districts_df = (districts_df["index"].str.split("–", expand=True).rename(
      {
          0: "district_number",
          1: "district_name"
      }, axis=1).join(districts_df).drop("index", axis=1))
  # strip the whitespace from the columns
  districts_df["district_number"] = districts_df["district_number"].str.strip()

  # create a new column with the district number
  districts_df["district"] = (
      districts_df["district_number"].str.extract(pattern_2).astype(int))
  districts_df.drop("district_number", axis=1, inplace=True)

  districts_df["link"] = districts_df["district_number"].apply(
      lambda x: x.str.strip() if x.dtype == "object" else x)
  districts_df["link"] = districts_df["district_number"].apply(
      lambda x: f"{zurich_description_url}#s-{x}")

  return districts_df


def find_breed_match(
    input_breed: str,
    breeds_df: pd.DataFrame,
    scoring_functions: list[Callable[[str, str], int]],
    scoring_threshold: int = 85,
) -> Optional[str]:
  """
    Find the match for the breed in the FCI breeds dataframe.
    breeds_df dataframe must have both a breed_en and alt_names column.
    """
  # Initialize the maximum score and best match
  max_score = scoring_threshold
  best_match = None

  # Iterate over each row in the breeds dataframe
  for index, breed_row in breeds_df.iterrows():
    # Get the alternative names for the current breed
    alternative_names = breed_row["alt_names"]

    # Calculate the score for the input breed and each alternative name
    # using each scoring function, and take the maximum of these scores
    current_score = max(
        max(
            scoring_function(input_breed, alt_name)
            for scoring_function in scoring_functions)
        for alt_name in alternative_names)
    # If the current score is greater than the maximum score, update the
    # maximum score and best match
    if current_score > max_score:
      max_score = current_score
      best_match = breed_row["breed_en"]

    # If the maximum score is 100, we have a perfect match and can break
    # out of the loop early
    if max_score == 100:
      break

  # Return the best match
  return best_match


def apply_fuzzy_matching_to_breed_column(
    dataframe: pd.DataFrame,
    breed_column: str,
    fci_df: pd.DataFrame,
    scoring_functions: list[Callable[[str, str], int]],
    scoring_threshold: int = 85,
) -> pd.Series:
  """Apply fuzzy matching to the breed column in the dataframe."""

  return dataframe[breed_column].apply(lambda breed: find_breed_match(
      breed, fci_df, scoring_functions, scoring_threshold=scoring_threshold))


def get_line_plots(data, x, group_by, highlight_list=None, **kwargs):
  """
    Generates an overlaid plot from data, highlighting specified groups with distinct colors.
    """
  if highlight_list is None:
    highlight_list = []
  # Default highlight colors
  default_highlight_colors = [
      "#DC143C",  # Crimson Red
      "#4169E1",  # Royal Blue
      "#50C878",  # Emerald Green
      "#DAA520",  # Goldenrod
  ]

  plots = []
  colors = kwargs.get("colors",
                      ["gray" if not highlight_list else "lightgray"])
  highlight_colors = kwargs.get("highlight_colors", default_highlight_colors)

  # Extend the highlight_colors list if there are more highlighted groups than colors
  if len(highlight_list) > len(highlight_colors):
    highlight_colors = highlight_colors * (
        len(highlight_list) // len(highlight_colors) + 1)

  for i, group_value in enumerate(data[group_by].unique()):
    # Filter the DataFrame for the specified value
    filtered_data = data.query(f"{group_by} == @group_value")
    # Sort the data by x then group_by
    filtered_data = filtered_data.sort_values([group_by, x])

    # Determine the color for the plot
    plot_color = (highlight_colors[highlight_list.index(group_value)] if
                  group_value in highlight_list else colors[i % len(colors)])

    # Create a line plot for the specified value
    line_plot = filtered_data.hvplot(
        color=plot_color,
        x=x,
        by=group_by,
        alpha=0.9,
    )

    # Create a scatter plot for the specified value
    scatter_plot = filtered_data.hvplot.scatter(color=plot_color,
                                                x=x,
                                                by=group_by)

    # Combine the line plot and scatter plot
    plot = line_plot * scatter_plot
    plots.append(plot)

  # Overlay the plots
  combined_plot = hv.Overlay(plots).opts(active_tools=["box_zoom"])

  return combined_plot


@pn.depends(yearly_player.param.value)
@pn.cache(max_items=10, policy="LRU")
def get_dog_age_butterfly_plot(roster):
  """
    Decorated with @pn.depends, this function generates a butterfly plot of male
    and female dog age distributions for a given roster year.

    Parameters:
    roster (int): The roster year to filter the dog data by.

    Returns:
    hvplot: A butterfly plot of male and female dog age distributions for the given roster year.
    """
  # Define bar plot options
  bar_opts = dict(
      invert=True,
      height=500,
      width=400,
      rot=90,
      xlim=(0, 24),
      xlabel="",
      yaxis="bare",
      ylabel="Count",
  )
  # Filter the DataFrame for the roster
  filtered_dog_data = pd.read_csv(PROCESSED_DOG_DATA_PATH)
  # filtered_dog_data = pd.read_csv("../data/processed_dog_data.csv")
  roster_dog_data = filtered_dog_data.query(f"roster=={roster}")
  # Filter for the is_male_dog
  male_roster_dog_data = roster_dog_data.loc[roster_dog_data["is_male_dog"]]
  male_roster_dog_data = (male_roster_dog_data.groupby(
      ["dog_age"]).size().reset_index(name="age_frequency"))
  male_roster_dog_data = male_roster_dog_data.set_index("dog_age")
  total_male = male_roster_dog_data["age_frequency"].sum()
  male_plot = male_roster_dog_data.hvplot.bar(
      **bar_opts,
      ylim=(0, 620),
      title=f"Male Dog Age Distribution || {roster} || {total_male} Canines",
      color="skyblue",
  ).opts(active_tools=["box_zoom"])

  female_roster_dog_data = roster_dog_data[~roster_dog_data["is_male_dog"]]
  female_roster_dog_data = (female_roster_dog_data.groupby(
      ["dog_age"]).size().reset_index(name="age_frequency"))
  female_roster_dog_data = female_roster_dog_data.set_index("dog_age")
  total_female = female_roster_dog_data["age_frequency"].sum()
  female_roster_dog_data["age_frequency"] = (
      -1 * female_roster_dog_data["age_frequency"])
  female_plot = female_roster_dog_data.hvplot.bar(
      **bar_opts,
      ylim=(-620, 0),
      title=
      f"Female Dog Age Distribution || {roster} || {total_female} Canines",
      color="pink",
  ).opts(active_tools=["box_zoom"])
  return (female_plot + male_plot).opts(shared_axes=False, )


@pn.depends(roster_button.param.value)
@pn.cache(max_items=10, policy="LRU")
def get_neighborhood_dog_density(roster):
  """Callback function to generate a choropleth map of dog density for a given roster year."""
  # Load and filter dog data
  df = pd.read_csv(PROCESSED_DOG_DATA_PATH)
  df = df.query(f"roster=={roster}")

  # Aggregate dog count by neighborhood
  df = (df.groupby([
      "neighborhood"
  ]).size().reset_index(name="total_dogs")).set_index("neighborhood")

  # Load neighborhood geospatial data
  map_gdf = gpd.read_file(NEIGHBORHOOD_GDF_PATH).set_index("neighborhood")

  # Merge dog count and geospatial data
  roster_dog_data_gdf = map_gdf.merge(df,
                                      left_index=True,
                                      right_index=True,
                                      how="left")

  # Calculate dog density
  roster_dog_data_gdf["dog_density"] = (roster_dog_data_gdf["total_dogs"] /
                                        roster_dog_data_gdf["area_km2"])

  # Create and return a choropleth map of dog density
  return gv.Polygons(roster_dog_data_gdf).opts(
      **poly_opts,
      color="dog_density",
      colorbar=True,
      tools=["hover", "tap", "box_select"],
      color_levels=6,
      title=f"Dog Density {roster} [dogs/km2]",
  )


@pn.depends(roster_button.param.value)
@pn.cache(max_items=10, policy="LRU")
def get_neighborhood_dogs_total(roster):
  """Callback function to generate a choropleth map of total dogs for a given roster year."""
  # Load and filter dog data
  df = pd.read_csv(PROCESSED_DOG_DATA_PATH)
  df = df.query(f"roster=={roster}")

  # Aggregate dog count by neighborhood
  df = df.groupby("neighborhood").size().reset_index(name="total_dogs")
  df = df.set_index("neighborhood")

  # Load neighborhood geospatial data
  map_gdf = gpd.read_file(NEIGHBORHOOD_GDF_PATH).set_index("neighborhood")

  # Merge dog count and geospatial data
  dog_gdf = map_gdf.merge(df, left_index=True, right_index=True, how="left")

  # Create and return a choropleth map of total dogs
  return gv.Polygons(dog_gdf).opts(
      **poly_opts,
      color="total_dogs",
      colorbar=True,
      tools=["hover", "tap", "box_select"],
      color_levels=6,
      title=f"Dogs Distribution || {roster}",
  )


@pn.depends(roster_button.param.value)
@pn.cache(max_items=10, policy="LRU")
def get_pop_neighborhood_count(roster):
  """Callback function to generate a choropleth map of total population for a given roster year."""
  # Load and filter population data
  df = pd.read_csv(PROCESSED_POP_DATA_PATH)
  df = df.query(f"roster=={roster}")

  # Aggregate population by neighborhood
  df = df.groupby("neighborhood")["pop_count"].sum()

  # Load neighborhood geospatial data
  map_gdf = gpd.read_file(NEIGHBORHOOD_GDF_PATH).set_index("neighborhood")

  # Merge population and geospatial data
  agg_gdf = map_gdf.merge(df, left_index=True, right_index=True, how="left")

  # Create and return a choropleth map of population count
  return gv.Polygons(agg_gdf).opts(
      **poly_opts,
      color="pop_count",
      colorbar=True,
      tools=["hover", "tap", "box_select"],
      title=f"Pop Count {roster}",
      aspect="equal",
      color_levels=6,
  )


@pn.depends(roster_button.param.value)
@pn.cache(max_items=10, policy="LRU")
def get_pop_density(roster):
  """Callback function to generate a choropleth map of population density for a given roster year."""
  # Load and filter population data
  df = pd.read_csv(PROCESSED_POP_DATA_PATH)
  df = df.query(f"roster=={roster}")

  # Aggregate population by neighborhood
  df = df.groupby("neighborhood")["pop_count"].sum()

  # Load neighborhood geospatial data
  map_gdf = gpd.read_file(NEIGHBORHOOD_GDF_PATH).set_index("neighborhood")

  # Merge population and geospatial data
  agg_gdf = map_gdf.merge(df, left_index=True, right_index=True, how="left")

  # Calculate population density
  agg_gdf["pop_density"] = agg_gdf["pop_count"] / agg_gdf["area_km2"]

  # Create and return a choropleth map of population density
  return gv.Polygons(agg_gdf).opts(
      **poly_opts,
      color="pop_density",
      colorbar=True,
      tools=["hover", "tap", "box_select"],
      title=f"Pop Density {roster} [persons/km2]",
      aspect="equal",
      color_levels=6,
  )


@pn.depends(roster_button.param.value, quantile_income_button.param.value)
@pn.cache(max_items=10, policy="LRU")
def get_income_polygon(roster, quantile):
  """Callback function to generate a choropleth map of median income for a given roster year."""
  df = pd.read_csv(PROCESSED_INCOME_DATA_PATH)
  df = df.query(f"roster == {roster}")
  df = df.set_index("neighborhood")

  # Load neighborhood geospatial data
  map_gdf = gpd.read_file(NEIGHBORHOOD_GDF_PATH).set_index("neighborhood")
  # Merge income and geospatial dataframes

  income_gdf = map_gdf.merge(df, left_index=True, right_index=True, how="left")

  income_cmap = list(
      sns.color_palette("light:" + "#388E3C", n_colors=5).as_hex())
  quantile_title = quantile.replace("_", " ").title()
  return gv.Polygons(income_gdf).opts(
      **poly_opts,
      color=f"{quantile}",
      cmap=income_cmap,
      clim=(30, 100),
      colorbar=True,
      tools=["hover", "tap", "box_select"],
      title=f"{quantile_title} in a thousand fracs | {roster}",
      aspect="equal",
  )


def train_and_predict_arima(data, end_year, n_periods=1):
  """Takes in a Series and returns a forecast using auto_arima."""
  end_year = pd.to_datetime(end_year, format="%Y")
  train = data.loc[data.index < end_year]
  model = auto_arima(
      train,
      seasonal=False,
      trace=False,
      error_action="ignore",
      suppress_warnings=True,
  )
  return pd.DataFrame(model.predict(n_periods=n_periods), columns=[data.name])


def forecast_arima(data, end_year, n_periods=1, model_desc="Model"):
  """Takes in a DataFrame and returns a DataFrame with n forecast for each column."""
  forecasts = []
  for col in tqdm(data.columns, desc=f"Training {model_desc} {end_year}"):
    forecasts.append(
        train_and_predict_arima(data[col], end_year, n_periods=n_periods))

  return pd.concat(forecasts, axis=1)


def _embeddings(args):
  """Generate UMAP embeddings for a given n_neighbors and min_dist value."""
  df, neighbor, min_distance = args
  reducer = umap.UMAP(n_neighbors=neighbor, min_dist=min_distance)
  embeddings = reducer.fit_transform(df)
  return (neighbor, min_distance), embeddings


def compute_embeddings(df, n_neighbors_values, min_dist_values):
  """
    Compute UMAP embeddings in parallel for various n_neighbors and min_dist values.
    """
  with cf.ProcessPoolExecutor() as executor:
    # Compute embeddings in parallel
    embeddings_dict = dict(
        executor.map(
            _embeddings,
            [(df, n, d)
             for n, d in it.product(n_neighbors_values, min_dist_values)],
        ))

  return embeddings_dict


def compute_kmeans_labels(data, n_clusters):
  """Compute K-means cluster labels for the given data."""
  kmeans = KMeans(n_clusters=n_clusters, random_state=628).fit(data)
  return kmeans.labels_


def create_clustered_data_df(data, cluster_labels, name_2_columns=("x", "y")):
  """Create a DataFrame from the data and cluster labels."""
  # Convert data to a DataFrame if it's not already
  if not isinstance(data, pd.DataFrame):
    data_df = pd.DataFrame(data, columns=[i for i in name_2_columns])
  else:
    data_df = data.copy()

  # Add the cluster labels as a new column
  data_df["cluster"] = cluster_labels
  return data_df


def add_columns(data_df, other_df, columns):
  """Add specified columns from another DataFrame to the given DataFrame."""
  data = data_df.copy()
  for column in columns:
    data[column] = other_df[column].copy()
  return data


def create_scatterplot_with_origin_cross(data,
                                         x="x",
                                         y="y",
                                         title="Scatter Title"):
  """Create a scatter plot from the embeddings DataFrame."""
  plot = data.hvplot.scatter(
      x=x,
      y=y,
      by="cluster",
      hover_cols=["all"],
      size=10,
      width=500,
      height=300,
      legend=False,
      xaxis="bare",
      yaxis="bare",
      title=title,
  ).opts(tools=["box_zoom", "tap"], active_tools=["box_zoom"])
  v_line = hv.VLine(0).opts(color="gray", line_dash="dotted")
  h_line = hv.HLine(0).opts(color="gray", line_dash="dotted")
  return plot * v_line * h_line


def calculate_clusters_scores(embeddings_dict, cluster_range=None):
  """Calculate K-means clustering scores for all embeddings in the dictionary and return a DataFrame."""
  # Create a list to store the dataframes
  score_dataframes = []
  if cluster_range is None:
    # Define the range of cluster numbers
    cluster_range = list(range(2, 20))

  for embedding_key, embeddings in embeddings_dict.items():
    for num_clusters in cluster_range:
      kmeans_model = KMeans(n_clusters=num_clusters,
                            random_state=628).fit(embeddings)
      cluster_labels = kmeans_model.labels_

      silhouette_score = round(
          metrics.silhouette_score(embeddings, cluster_labels), 3)
      calinski_harabasz_score = round(
          metrics.calinski_harabasz_score(embeddings, cluster_labels), )
      davies_bouldin_score = round(
          metrics.davies_bouldin_score(embeddings, cluster_labels), 3)

      # Create a DataFrame
      scores_df = pd.DataFrame({
          "embedding_key": [embedding_key],
          "num_clusters": [num_clusters],
          "silhouette_score": [silhouette_score],
          "calinski_harabasz_score": [calinski_harabasz_score],
          "davies_bouldin_score": [davies_bouldin_score],
      })

      # Append the dataframe to the list
      score_dataframes.append(scores_df)

  # Concatenate all dataframes in the list into a single dataframe
  result_df = pd.concat(score_dataframes, ignore_index=True)

  return result_df
