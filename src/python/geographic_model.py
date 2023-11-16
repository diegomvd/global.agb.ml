"""
This script contains the training of an ABD model using solely geopgraphic predictors: latitude, longitude and elevation. 
"""

import numpy as np
import pandas as pd
# import geopandas as gpd
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold,cross_val_score
# from sklearn.pipeline import make_pipeline
# from dataset_creation import add_feature_from_raster

# data = gpd.read_file("/home/dibepa/git/global.agb.ml/data/training/tmp_preprocessed/tallo_learning_preprocessed_spatial_under_1Mg.csv/tallo_learning_preprocessed_spatial_under_1Mg.shp")

# data = add_feature_from_raster(
#     data,
#     "elev",
#     "/home/dibepa/git/global.agb.ml/data/training/raw/elevation/wc2.1_2.5m_elev.tif",
#     "float")

# print(data.columns)

# data["lon"] = data.geometry.x
# data["lat"] = data.geometry.y
# data = data[["abd","lon","lat"]]

# data.to_csv("/home/dibepa/git/global.agb.ml/data/training/tmp_preprocessed/test_xgbr_lon_lat.csv")

data = pd.read_csv("/home/dibepa/git/global.agb.ml/data/training/tmp_preprocessed/test_xgbr_lon_lat.csv")

print(data.columns)

y = np.array(data["abd"])
X = np.array(data[["lat","lon"]])

# X = data[['twq', 'yp', 'bgr1', 'bgr2', 'bgr3', 'ps', 'ts', 'ndvism_mea', 'ndviw', 'wavai']]
# y = data["abd"]

rkf = RepeatedKFold(n_splits=10)

xgbr = XGBRegressor(
            n_estimators=100,
            learning_rate = 0.04,
            max_depth = 6,
            min_child_weight = 3,
            subsample = 0.7,
            eval_metric = "rmse",
            objective='reg:squarederror'
            )

scores = cross_val_score(
            estimator= xgbr,
            X  = X,
            y = y,
            cv = rkf,
            scoring = "neg_mean_absolute_percentage_error"
        )

print(np.mean(-1*scores)*100)
