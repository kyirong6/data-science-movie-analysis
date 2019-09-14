# CMPT 353 - Movie Project

## Required Libraries
* string
* os
* Pandas
* Numpy
* Matplotlib
* SciPy
* keras
* tensorflow
* scikit-learn
* statsmodels

## Running The Project

### First Steps
1. Clone the repository
2. Install the required libraries

`conda install pandas, numpy, matplotlib, scipy, keras, tensorflow, scikit-learn, statsmodels `

### Data ETL
The repository comes with all the data required for the project as json.gz files in the data directory

Data is read from the json.gz files by importing the `data_etl.py` file and creating an instance of the `Data` class like so:

    data_manager = Data()

Creating an instance of the Data class will read in the data. To get the Pandas DataFrames, it's a simple method call

    merged, wikidata, omdb, rt = data_manager.get_all_data()

The `get_all_data` method call gives you a merged DataFrame of the wikidata, OMDb, and rotten tomatoes data. It also returns individual dataframes for the wikidata, OMDb, and rotten tomatoes. 

### Running the Statistical Analysis
To run the statistical analysis, simply run the command `python3 stats_analysis.py`. This will run the code for the statistical analysis on the movie data and produce graphs summarising the findings of the project.

### Running the Machine Learning Methods
To run the machine learning methods, simply run the command `python3 stats.py`. This will run the code for the machine learning methods on the movie data and produce graphs summarising the findings of the project.

This project was completed with a partner.
