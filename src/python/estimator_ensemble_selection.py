import pandas as pd
import geopandas as gpd
from dataset_creation import add_feature_from_raster, add_feature_from_polygon_layer

from xgboost import XGBRegressor
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
from pathlib import Path
import pickle

import numpy as np

from itertools import product

create_data = False

data_save_path = "/home/dibepa/git/global.agb.ml/data/map_validation/estimator_selection_data.csv"

if create_data: 

    df = pd.read_csv("/home/dibepa/git/global.agb.ml/data/map_validation/FOS_Plots_v2019.04.10_ABD_validation.csv")

    df = df[ ["Lat_cnt","Lon_cnt","AGB_local","AGB_Feldpausch","AGB_Chave"] ]
    df["geometry"] = list(zip(df.Lat_cnt, df.Lon_cnt))
    df = df.assign(abd_real=df[["AGB_local","AGB_Feldpausch","AGB_Chave"]].mean(axis=1))

    df = df[ ["geometry","abd_real"] ]
    df = df.groupby("geometry").mean().reset_index()

    df[ ['lat','lon'] ] = df['geometry'].apply(pd.Series)
    df = df.drop('geometry',axis='columns')

    points_gdf = gpd.GeoDataFrame(
        df,
        geometry= gpd.points_from_xy(x=df.lon, y=df.lat)
    )
    points_gdf = points_gdf.drop(['lat','lon'],axis='columns')

    print("Starting sampling of bioclimatic variables.")

    yearly_temperature              = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_annual_mean_temp.tif"
    dry_quarter_temperature         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_driest_quarter.tif"
    wet_quarter_temperature         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_wettest_quarter.tif"
    cold_quarter_temperature        = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_coldest_quarter.tif"
    yearly_precipitation            = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_annual_precipitation.tif"
    wet_month_precipitation         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_wettest_month.tif"
    potential_evapotranspiration    = "/home/dibepa/git/global.agb.ml/data/training/raw/bioclimatic_data/et0_v3_yr.tif"
    temperature_seasonality         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_temperature_seasonality.tif"
    precipitation_seasonality       = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_seasonality.tif"

    bioclim_data = [
        ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_isothermality.tif","iso"),
        ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_max_temp_warmest_month.tif","mtwm"),
        ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_diurnal_range.tif","mdr"),
        ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_warmest_quarter.tif","mtwq"),
        ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_min_temp_coldest_month.tif","mtcm"),
        ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_coldest_quarter.tif","pcq"),
        ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_driest_month.tif","pdm"),
        ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_driest_quarter.tif","pdq"),
        ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_warmest_quarter.tif","pwaq"),
        ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_wettest_quarter.tif","pweq"),
        ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_temperature_annual_range.tif","tar"),
        (yearly_temperature,"yt"),
        (dry_quarter_temperature,"tdq"),
        (wet_quarter_temperature,"twq"),
        (cold_quarter_temperature,"tcq"),
        (yearly_precipitation,"yp"),
        (wet_month_precipitation,"pwm"),
        (potential_evapotranspiration,"pet"),
        (temperature_seasonality,"ts"),
        (precipitation_seasonality,"ps"),
    ]

    points_gdf.crs = "EPSG:4326"

    for path,col in bioclim_data:
        points_gdf = add_feature_from_raster(points_gdf, col, path, "float")
        print(points_gdf)

    points_gdf = add_feature_from_polygon_layer(points_gdf,"REALM","/home/dibepa/git/global.agb.ml/data/training/raw/biome_data/Ecoregions2017.shp","bgr")
    print(points_gdf)

    df = pd.DataFrame(points_gdf)

    print("Saving")
    df.to_csv(data_save_path)

else:

    df = pd.read_csv(data_save_path)    



class BGRBinaryEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self
    def fit(self, X , y=None):
        return self
    def transform(self, X):
        df = X.copy()
        # print(df)
        df["bgr_tuple"] = [binary_encoding(t) for t in df.bgr] 
        df["bgr1"] , df["bgr2"], df["bgr3"] = df.bgr_tuple.str
        df = df.drop(["bgr_tuple","bgr"],axis="columns")
        return df
    
# custom transformer for sklearn pipeline
class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return X[self.cols]

    def fit(self, X, y=None):
        return self    
    
class LogarithmizeWaterObservables(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X , y=None):
        return self
    def transform(self, X):
        df = X.copy()
        log_preds = log_predictors(self.cols)
        df[log_preds] = np.log(df[log_preds]+1)     
        return df    

def keep_estimator(elist,ename,eval):
    if eval>0.5:
        elist.append(ename)
    return elist


def build_estimator_list(estimator_dict):
    elist = []
    for estimator in estimator_dict:
        elist = keep_estimator(elist,estimator,estimator_dict[estimator])
    return elist    

def func(x):
    return np.log(x+1)

def inverse_func(x):
    return np.exp(x)-1

def binary_encoding(bgr):
    match bgr:
        case "Palearctic":
            return (0,0,0)
        case "Indomalayan":
            return (0,1,0)
        case "Australasia":
            return (0,0,1)
        case "Nearctic":
            return (0,1,1)
        case "Afrotropic":
            return (1,0,0)
        case "Neotropic":
            return (1,1,0)                    

log_transformer = FunctionTransformer(func=func,inverse_func=inverse_func)

def log_predictors(predictor_list):
    all = ["yp","pwm","pet","ps","pcq","pdq","pwaq","pweq"]
    return [f for f in all if f in predictor_list]


def predictor_list(predictor_string):
    pred = predictor_string.split("-")
    return pred 


pickled_model = "/home/dibepa/git/global.agb.ml/data/training/final_onlybioclim_with_notree/abd_model_onlybioclim_notrees.pkl"
with open(pickled_model, 'rb') as f:
    model = pickle.load(f)

min_error = 10000
best_estimators = []

# for eid in product([0,1], repeat = 10):

#     estimator_list = build_estimator_list(eid)

#     estimators = [ model.estimators_[i] for i in estimator_list ]

#     try:
#         predictions = np.array([ e.predict(df) for e in estimators])

#         y = predictions.mean(axis=0)

#         error = np.mean( np.abs( (df["abd_real"] - y) / df['abd_real'] ) )*100

#         if(error<min_error):
#             min_error = error
#             best_estimators = estimator_list        
    
#     except:
#         continue    

# print(min_error)
# print(estimator_list)


import seaborn as sns
from matplotlib import pyplot as plt

y = model.predict(df)
df['error_abs'] = np.abs( (df["abd_real"] - y) / df['abd_real'] ) * 100 
df['error_sign'] = (df["abd_real"] - y) / df['abd_real'] * 100 
df['diff_abs'] =  np.abs((df["abd_real"] - y)) 
df['diff_sign'] = (df["abd_real"] - y) 
print(df)

sns.histplot(data=df,x="error_abs",kde=True)
plt.show()
sns.histplot(data=df,x="error_sign",kde=True)
plt.show()

sns.histplot(data=df,x="diff_abs",kde=True)
plt.show()
sns.histplot(data=df,x="diff_sign",kde=True)
plt.show()
