"""
Script to estimate errors of the grouping estimation of ABD with respect to GEDI ABD estimates. 
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def absolute_error(x,y):
    return np.abs(x-y)

def signed_difference(x,y):
    return x-y

def relative_error(x,y):
    return 100*signed_difference(x,y)/x

def absolute_percentual_error(x,y):
    return 100*absolute_error(x,y)/x

def mae(x,y):
    return np.mean(absolute_error(x,y))

def msd(x,y):
    return np.mean(signed_difference(x,y))

def mape(x,y):
    return np.mean(absolute_percentual_error(x,y))




africa = gpd.read_file("/home/dibepa/git/global.agb.ml/data/training_set_validation/Africa.shp")
asia = gpd.read_file("/home/dibepa/git/global.agb.ml/data/training_set_validation/Asia.shp")
asia_insular = gpd.read_file("/home/dibepa/git/global.agb.ml/data/training_set_validation/Asia (insular).shp")
south_america = gpd.read_file("/home/dibepa/git/global.agb.ml/data/training_set_validation/South America.shp")
north_america = gpd.read_file("/home/dibepa/git/global.agb.ml/data/training_set_validation/North America.shp") 
australia = gpd.read_file("/home/dibepa/git/global.agb.ml/data/training_set_validation/Australia.shp")
new_zealand = gpd.read_file("/home/dibepa/git/global.agb.ml/data/training_set_validation/New Zealand.shp")

africa = africa[["agb","gediAGBD"]]
asia = asia[["agb","gediAGBD"]]
asia_insular = asia_insular[["agb","gediAGBD"]]
south_america = south_america[["agb","gediAGBD"]]
north_america = north_america[["agb","gediAGBD"]]
australia = australia[["agb","gediAGBD"]]
new_zealand = new_zealand[["agb","gediAGBD"]]

africa["continent"] = "Africa"
asia["continent"] = "Asia"
asia_insular["continent"] = "Asia (insular)"
south_america["continent"] = "South America"
north_america["continent"] = "North America"
australia["continent"] = "Australia"
new_zealand["continent"] = "New Zealand"

data = pd.concat( [africa,asia,asia_insular,south_america,north_america,australia,new_zealand], axis = 0 ).reset_index()

data["abs_error"] = absolute_error(data["gediAGBD"],data["agb"])
data["sign_diff"] = signed_difference(data["gediAGBD"],data["agb"])
data["relative_error"] = relative_error(data["gediAGBD"],data["agb"])
data["percent_error"] = absolute_percentual_error(data["gediAGBD"],data["agb"])

print( data.groupby("continent").mean() ) 
filt_data = data[(data.agb < 1000) & (data.agb > 10)]
print( filt_data.groupby("continent").mean() )

data_mean = filt_data.groupby("continent").mean()

sns.relplot( filt_data, x= "agb", y="abs_error", hue="continent", kind= "scatter" )
plt.yscale("log")
plt.xscale("log")

sns.catplot( data_mean, x="continent", y="percent_error", size=10 )
plt.show()
