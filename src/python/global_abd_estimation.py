#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for the estimation of aboveground biomass density at the global level.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
from matplotlib import pyplot as plt
from dataset_creation import add_feature_from_raster, add_feature_from_polygon_layer
from allometry import estimate_tree_biomass_jucker2017, estimate_tree_crown_radius_jucker2017, estimate_tree_height_jucker2017
from instance_grouping import group_by_category,build_aggfunc_dict,aggregate_values_by_group
from abd_estimation import estimate_biomass_density

###
# Import tree-size database as GeoDataFrame.
###

# df = pd.read_csv("./tree_data/Tallo.csv")
# tallo = gpd.GeoDataFrame(
#     df, 
#     geometry=gpd.points_from_xy(df.longitude, df.latitude)
# )
# tallo.crs = "EPSG:4326"

# # tallo = tallo.head(10)

# # Select which columns to keep: tree size metrics and functional group for AGB calculation, tree id for traceability and geometry.
# columns = ["stem_diameter_cm", "height_m", "crown_radius_m","tree_id","division","geometry"]
# tallo = tallo[columns].reset_index(drop=True)

# # Rename columns.
# tallo = tallo.rename(
#     columns = {
#         "stem_diameter_cm" : "d",
#         "height_m" : "h",
#         "crown_radius_m" : "cr",
#         "tree_id" : "tid"
#     }
# )

# print(tallo)

# ###
# # Include biome and realm data.
# ###

# # Vector data on evolutionary history and biome type for allometry specification.
# biomes_and_biogeographic_realms = "./biome_data/Ecoregions2017.shp"

# tallo = add_feature_from_polygon_layer(tallo,"BIOME_NUM",biomes_and_biogeographic_realms,"bid")    
# tallo = add_feature_from_polygon_layer(tallo,"REALM",biomes_and_biogeographic_realms,"bgr") 

# print(tallo)

# tallo.to_file("./revised_model/temp_data/tallo_bibgr.shp",driver="ESRI Shapefile")

###
# Fill no-data on height and crown radius using Jucker et al. 2017 D allometry.
###

# tallo = gpd.read_file("./revised_model/temp_data/tallo_bibgr.shp")

# # Replace -9999.9999 value that indicates no-data with Nan.
# tallo = tallo.replace(-9999.9999, np.nan)
# tallo = tallo.replace("-9999.9999", np.nan)

# # Estimate CR and D

# tallo["cr"] = np.where(
#     tallo['cr'].isna(),
#     [estimate_tree_crown_radius_jucker2017(row[0],row[1],row[2],row[3],row[4]) for row in zip(tallo["bgr"],tallo["bid"], tallo["division"], tallo["d"],tallo["h"])],
#     tallo['cr']
# )
# print(tallo)
# tallo["h"] = np.where(
#     tallo['h'].isna(),
#     [estimate_tree_height_jucker2017(row[0],row[1],row[2],row[3],row[4]) for row in zip(tallo["bgr"],tallo["bid"], tallo["division"], tallo["d"],tallo["cr"])],
#     tallo['h']
# ) 

# print(tallo)

# tallo.to_file("./revised_model/temp_data/tallo_bibgr_fullhcr.shp",driver="ESRI Shapefile")

# ###
# # Calculate AGB for every tree.
# ###

# tallo["agb"] = [estimate_tree_biomass_jucker2017(row[0],row[1],row[2]) for row in zip(tallo["division"],tallo["cr"],tallo["h"])]

# print(tallo)

# tallo.to_file("./revised_model/temp_data/tallo_bibgr_fullhcr_agb.shp",driver="ESRI Shapefile")

###
# Add data on biomass predictors to the tree database.
###

# tallo = gpd.read_file("./revised_model/temp_data/tallo_bibgr_fullhcr_agb.shp")
# # print(tallo["agb"].mean())
# # print(tallo["agb"].max())
# # print(tallo["agb"].min())

# # Raster data on bioclimatic predictors.
# yearly_temperature              = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_annual_mean_temp.tif"
# dry_quarter_temperature         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_driest_quarter.tif"
# wet_quarter_temperature         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_wettest_quarter.tif"
# cold_quarter_temperature        = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_coldest_quarter.tif"
# yearly_precipitation            = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_annual_precipitation.tif"
# wet_month_precipitation         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_wettest_month.tif"
# potential_evapotranspiration    = "./bioclimatic_data/et0_v3_yr.tif"
# aridity_index                   = "./bioclimatic_data/ai_v3_yr.tif"
# temperature_seasonality         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_temperature_seasonality.tif"
# precipitation_seasonality       = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_seasonality.tif"

# # Raster data on tree density. 
# tree_density                    = "./tree_density_data/tree_density_biome_based_model_crowther_nature_2015_4326_float32.tiff"

# tree_cover                        = "/home/dibepa/Documents/GLADMeanTreeCover/glad_scrapping_builder/resampled_and_merged_30s_1km/resampled_30s_1km_global.tif" 

# # Join raster data paths and sample them
# raster_data = [
#     (yearly_temperature,"yt"),
#     (dry_quarter_temperature,"tdq"),
#     (wet_quarter_temperature,"twq"),
#     (cold_quarter_temperature,"tcq"),
#     (yearly_precipitation,"yp"),
#     (wet_month_precipitation,"pwm"),
#     (potential_evapotranspiration,"pet"),
#     (aridity_index,"ai"),
#     (temperature_seasonality,"ts"),
#     (precipitation_seasonality,"ps"),
#     (tree_density,"dens"),
#     (tree_cover,"tcov") 
# ]

# for path,col in raster_data:
#     tallo = add_feature_from_raster(tallo, col, path, "float")
#     print(tallo)

# tallo.to_file("./revised_model/temp_data/tallo_bibgr_fullhcr_agb_bioclim.shp",driver="ESRI Shapefile")

###
# Group trees.
###

# tallo = gpd.read_file("./temp_data/tallo_bibgr_fullhcr_agb_bioclim.shp")

# # tallo = tallo.head(5000)

# # To remove possible nans in bioclimatic variables.
# tallo = tallo.replace(-9999.9999, np.nan)
# tallo = tallo.replace("-9999.9999", np.nan)
# tallo = tallo.replace(-9999.00, np.nan)
# tallo = tallo.replace("-9999.00", np.nan)

# print(tallo)

# # Grouping parameters.

# # Buffer radius in meters.
# rmin = 500
# rmax = 25000
# dr = 500

# # Bounding box threshold.
# bboxl = 50000

# # Tile maximum side length.
# lmax = 50000 
# dl = 500

# # Sample threshold.
# sample_thr = 50

# tallo, grouping_params = group_by_category(tallo,"bid",rmin,rmax,dr,lmax,dl,bboxl,"tid",sample_thr,False,"./temp_data/grouping2/tallo_2_")

# tallo.to_file("./temp_data/tallo_bibgr_fullhcr_agb_bioclim_grouped_2.shp",driver="ESRI Shapefile")
# grouping_params.to_csv("./temp_data/grouping_params_2.csv")

# # tallo = gpd.read_file("./temp_data/tallo_bibgr_fullhcr_agb_bioclim_grouped.shp")


# print(tallo)
# print(tallo.shape)
# print(tallo.gid.unique().shape)
# print(np.mean(tallo.n_samples))
# print(np.min(tallo.n_samples))
# print(np.max(tallo.n_samples))
# print(tallo[tallo.n_samples < 50].shape)
# print(tallo[tallo.n_samples > 50].shape)

# print()
# print(tallo[tallo.n_samples<50].gid.unique().shape[0])
# print(tallo[tallo.n_samples>=50].gid.unique().shape[0])
# print()

###
# Estimate ABD in each group. 
###
# print("Starting")
# tallo = gpd.read_file("./temp_data/tallo_bibgr_fullhcr_agb_bioclim_grouped.shp")
# # tallo = tallo.replace(-9999.9999, np.nan)
# # tallo = tallo.replace("-9999.9999", np.nan)
# # tallo = tallo.replace(-9999, np.nan)
# # tallo = tallo.replace("-9999", np.nan)
# tallo["dens"] = np.where(
#     tallo["dens"] < -9000.0,
#     np.nan,
#     tallo["dens"]
# )
# tallo["lnagb"] = np.log(tallo["agb"])
# print(tallo)

# # tallo = tallo.head(100)

# # rng = np.random.default_rng()
# # seed = rng.integers(np.iinfo(np.int32).max)

# kdeparams = {
#     "niter" : 100,
#     "bwmin" : 0.2,
#     "bwmax" : 1.5,
#     "kernel": "gaussian",
#     "in_splits" : 5,
#     "out_splits" : 10
# }

# # kg per km2
# abd = estimate_biomass_density(tallo,"lnagb","dens",kdeparams,10,True,"./temp_data/abd_per_group_temp.csv")
# print(abd)
# abd.to_csv("./temp_data/abd_per_group.csv")


# print(tallo[tallo.isna()].size)

# groups = tallo.gid.unique()
# tallo = tallo[["bid","bgr","gid","agb"]]
# for g in groups:
#     tallog = tallo[tallo.gid==g]
#     biome = tallog.bid.head(1)
#     bgr = tallog.bgr.head(1)
#     print((biome,bgr))
#     sns.histplot(data=tallog, x="agb", log_scale = True)
#     sns.ecdfplot(data=tallog, x="agb", log_scale = True)
#     plt.show()

###
# Aggregate values in each group and store the data for verification. 
###

# abd = pd.read_csv("./temp_data/abd_per_group.csv")

# # Removing locations with a null tree density. Probably due to raster mis-alignement.
# abd = abd.drop(abd[abd.td_mean <= 0.0].index)

# # Change Standard deviations to coefficient of variation to allow comparison.
# abd["abd_cv"] = abd["std_abd"]/abd["mean_abd"]
# abd["td_cv"] = abd["td_std"]/abd["td_mean"]
# abd = abd.drop(["std_abd","td_std"],axis = "columns")

# print(abd)
# abd.to_csv("./temp_data/abd_per_group_nonan.csv")

# # Import tallo database and filter by group.
# tallo = gpd.read_file("./temp_data/tallo_bibgr_fullhcr_agb_bioclim_grouped.shp")

# # tree_cover = "/home/dibepa/Documents/GLADMeanTreeCover/glad_scrapping_builder/resampled_and_merged_30s_1km/resampled_30s_1km_global.tif" 
# # tallo = add_feature_from_raster(tallo, "tcov", tree_cover, "float")
# # print(tallo)

# # Add NDVI by season
# ndvi_data = [
#     ("./ndvi_data/ndvi_1999_2019_summer.tif", "ndvism"),
#     ("./ndvi_data/ndvi_1999_2019_fall.tif", "ndvif"),
#     ("./ndvi_data/ndvi_1999_2019_winter.tif", "ndviw"),
#     ("./ndvi_data/ndvi_1999_2019_spring.tif", "ndvisp") 
# ]

# for data, name in ndvi_data:
#     tallo = add_feature_from_raster(tallo, name, data, "float")

# # Add P-PET
# tallo["wavai"] = tallo["yp"]-tallo["pet"]

# tallo.to_file("./temp_data/tallo_bibgr_fullhcr_agb_bioclim_grouped.shp",driver="ESRI Shapefile")


# Calculate centroid of each group to allow for spatial cross validation.

# tallo = gpd.read_file("./temp_data/tallo_bibgr_fullhcr_agb_bioclim_grouped.shp")

# tallo_filt = tallo[["geometry","gid"]]


# centroids = tallo_filt.dissolve(by='gid',aggfunc='first').to_crs('+proj=cea')

# print("Done. Calculating each group's centroid...")
# centroids["geometry"] = centroids.geometry.centroid

# centroids = centroids.to_crs(tallo.crs)

# print("Done.")
# print(centroids)

# centroids.to_file("./temp_data/group_centroids.shp", driver = "ESRI Shapefile")

# # Aggregate predictors by group membership: mean and coefficient of variation. 

# # Remove the geometry information.
# tallo = pd.DataFrame(tallo.drop("geometry",axis="columns"))

# # Remove tree size metrics, tree id and division.
# tallo = tallo.drop(["d","h","cr","tid","division"],axis="columns")
# print(tallo)

# # Create the dictionary with aggregation functions.

# aggfunc_dict = {}

# # Columns to get mean and standard deviation: 
# agg_cols = ["agb","yt","tdq","twq","tcq","yp","pwm","pet","ai","ts","ps","dens","tcov","ndvism","ndvif","ndviw","ndvisp","wavai"]

# # Columns to get first value (same value for all the group):
# id_cols = ["n_samples","bgr","bid"]

# aggparams_list = [
#     (id_cols, "first",False),
#     (agg_cols,"mean",True),
#     (agg_cols,"std",True)
# ]

# tallo_agg = aggregate_values_by_group(tallo,"gid",aggparams_list)

# # Replace the standard deviation by the Coefficient of Variation.
# std_cols = []
# for col in agg_cols:
#     cstd = col+"_std"
#     cmean = col+"_mean"
#     tallo_agg[col+"_cv"] = np.abs(tallo_agg[cstd]/tallo_agg[cmean])
#     std_cols.append(cstd)    

# tallo_agg = tallo_agg.drop(std_cols,axis="columns")

# print(tallo_agg)
# tallo_agg.to_csv("./temp_data/grouped_predictors.csv")

# ####
# # Create the learning dataset.
# ####

# tallo_agg = pd.read_csv("./temp_data/grouped_predictors.csv")
# abd = pd.read_csv("./temp_data/abd_per_group_nonan.csv")


# # Filter dataframes to keep only relevant columns for the learning process. 
# abd = abd[["gid","mean_abd"]]
# tallo_agg = tallo_agg[["gid","bgr","yt_mean","tdq_mean","twq_mean","tcq_mean","yp_mean","pwm_mean","pet_mean","ai_mean","ts_mean","ps_mean","dens_mean","tcov_mean","ndvism_mean","ndvif_mean","ndviw_mean","ndvisp_mean","wavai_mean","n_samples"]]

# tallo_learn = abd.join(tallo_agg.set_index('gid'),on="gid")
# print(tallo_learn)
# tallo_learn.to_csv("./temp_data/tallo_learning.csv")

####
# Add spatial information: group centroids.
####

# group_centroids = gpd.read_file("./temp_data/group_centroids.shp")
# tallo = pd.read_csv("./temp_data/tallo_learning.csv")
# print(tallo)
# tallo_learn = gpd.GeoDataFrame(tallo.join(group_centroids.set_index("gid"), on="gid")).drop("Unnamed: 0", axis="columns")
# print(tallo_learn)
# tallo_learn.to_file("./temp_data/tallo_learning_spatial.shp", driver = "ESRI Shapefile")

####
# Feature engeneering
####

# tallo_learn = pd.read_csv("./temp_data/tallo_learning.csv")
tallo_learn = gpd.read_file("./temp_data/tallo_learning_spatial.shp")

# Transform Biomass Density from kg/km² to Mg/ha: x10^-5.
tallo_learn["mean_abd"] = tallo_learn["mean_abd"]*np.power(10.0,-5.0)


# Remove Biomasses of less than 2 Mg/ha and more than 2000 Mg/ha.
tallo_learn = tallo_learn.drop(tallo_learn[tallo_learn.mean_abd > 1000.0].index)
tallo_learn = tallo_learn.drop(tallo_learn[tallo_learn.mean_abd < 2.0].index)

sns.histplot(data=tallo_learn, x="n_samples", log_scale = True)
sns.histplot(data=tallo_learn, x="mean_abd", log_scale = True)
plt.show()

sns.scatterplot(data=tallo_learn, x="n_samples", y="mean_abd")
plt.xscale("log")
plt.yscale("log")
plt.show()

# Log-transform data with heavy-tail distributions: Biomass density and precipitation related predictors can be log-transformed. This has an impact in Random Forest regression.
log_cols = ["mean_abd","yp_mean","pwm_mean","pet_mean","ai_mean","ps_mean"]
for col in log_cols:
    tallo_learn[col] = np.log(tallo_learn[col])

# Binary encoding of Biogeographic Realm.

# Binary encoding of biogeographical realm:
# Palearctic   = 0 0 0
# Indomalayan = 0 1 0
# Australasia = 0 0 1
# Nearctic    = 0 1 1
# Afrotropic  = 1 0 0
# Neotropic   = 1 1 0

print(tallo_learn.bgr.value_counts())
# Drop Oceania biogeographical realm as there is a single sample.
tallo_learn = tallo_learn.drop(tallo_learn[tallo_learn.bgr=="Oceania"].index).reset_index(drop=True)

tallo_learn["bgr1"] = np.nan
tallo_learn["bgr2"] = np.nan
tallo_learn["bgr3"] = np.nan

tallo_learn["bgr1"] = np.where(
                (tallo_learn["bgr"] == "Afrotropic") | (tallo_learn["bgr"] == "Neotropic"),
                 1, 0 )
tallo_learn["bgr2"] = np.where(
                (tallo_learn["bgr"] == "Indomalayan") | (tallo_learn["bgr"] == "Nearctic") | (tallo_learn["bgr"] == "Neotropic"),
                 1, 0 )
tallo_learn["bgr3"] = np.where(
                (tallo_learn["bgr"] == "Nearctic") | (tallo_learn["bgr"] == "Australasia"),
                 1, 0 )

# Calculate sample weights based on the number of sampled trees in each group:
# This is an open question. Is it more pertinent to put sample weights or to interpret the number of samples as a potential confidence measure? If sample weights should be put, probably a saturating function of the number of sampled trees is the best option. 
def weights(ns,k):
    return (ns/(ns+k))

tallo_learn["wsample"] = tallo_learn["n_samples"].apply(lambda x: weights(x,50.0))

# Drop tree-density, group Id, non-encoded biogeogrpahic realm and number of samples for the learning dataset.
# tallo_learn = tallo_learn.drop(["dens_mean","bgr","Unnamed: 0","gid","n_samples"],axis="columns")
tallo_learn = tallo_learn.drop(["dens_mean","bgr","gid","n_samples"],axis="columns")

# Remove column suffix for simplicity.
tallo_learn = tallo_learn.rename(
   columns = { 
        "mean_abd" : "abd",
        "yt_mean" : "yt",
        "tdq_mean":"tdq",
        "twq_mean":"twq",
        "tcq_mean":"tcq",
        "yp_mean":"yp",
        "pwm_mean":"pwm",
        "pet_mean":"pet",
        "ai_mean":"ai",
        "ts_mean":"ts",
        "ps_mean":"ps",
        "tcov_mean":"tcov",
        "ndvism_mean":"ndvism",
        "ndvif_mean":"ndvif",
        "ndviw_mean":"ndviw",
        "ndvisp_mean":"ndvisp",
        "wavai_mean":"wavai"
    }
)

print(tallo_learn)
# tallo_learn.to_csv("./temp_data/tallo_learning_preprocessed.csv")
tallo_learn.to_file("./temp_data/tallo_learning_preprocessed_spatial_under_1Mg.csv", driver="ESRI Shapefile")

# Create 5 data folds for nested cross-validation.
# tallo_learn = pd.read_csv("./temp_data/tallo_learning_preprocessed.csv")

# rng = np.random.default_rng()
# id_array = tallo_learn.index.to_numpy()
# rng.shuffle(id_array)

# n_folds = 5
# splits_id = np.array_split(id_array,n_folds)

# for i in range(n_folds):
#     test = tallo_learn.iloc[splits_id[i]]
#     train = tallo_learn.drop(splits_id[i], axis = 0)
#     #print(train)
#     test.to_csv("./training_data_XGBR/tallo_learning_preprocessed_outcvloop_test_fold_" + str(i+1) + ".csv",index=False)
#     train.to_csv("./training_data_XGBR/tallo_learning_preprocessed_outcvloop_train_fold_" + str(i+1) + ".csv",index=False)