import geopandas as gpd
import pandas as pd
from natsort import natsorted
from pathlib import Path
import re
import numpy as np
import rasterio
from dataset_creation import add_feature_from_raster, add_feature_from_polygon_layer

# tree_density_file = "/home/dibepa/git/global.agb.ml/data/training/raw/tree_density/tree_density_biome_based_model_crowther_nature_2015_4326_float32.tiff" 

# predictor_range_dir = "/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/tmp"

# training_dataset = "/home/dibepa/git/global.agb.ml/data/training/tmp_preprocessed/tallo_learning_preprocessed_onlybioclim.csv"

# ##########################################################
# print("Starting building the predictor out of range dataframe")

# pfiles = natsorted( Path(predictor_range_dir).glob('predictors_in_range*.csv'), key=str )

# df_predictor = pd.DataFrame()
# for i,file in enumerate(pfiles):

#     # Remove Antarctica
#     window = re.findall('predictors_in_range(.*)', file.stem)[0] 
#     lat = float(window.split("_")[-1])
    
#     if lat>-60:

#         df = pd.read_csv(file)

#         df = df.drop( df[df.in_range > 98].index )

#         if len(df.index)>0:

#             df_predictor = pd.concat([df_predictor,df], axis = "rows")
    
#     print("Processed {} out of {} files.".format(i+1,len(pfiles)))

# df_predictor = df_predictor.reset_index(drop=True)
# df_predictor = df_predictor.astype({'in_range':'uint16','x':'float32','y':'float32'})


# ###########################################################
# print("Starting the creation of new 0 biomass locations")


# samples = 0
# points = pd.DataFrame() 
# df_training = pd.read_csv(training_dataset)

# tree_density = rasterio.open(tree_density_file)

# while len(points.index)<len(df_training.index):
    
#     s = df_predictor.sample()

#     point = (s.x,s.y)
#     id = s.index

#     df_predictor.drop(id)

#     td = [x for x in tree_density.sample([point])]

#     if td[0] == 0 :
#         p = pd.DataFrame({"x": s.x, "y": s.y})
#         points = pd.concat([points,p], axis = "rows")
#         samples +=1 
#         print("Sampled {} points".format(samples))

# points_gdf = gpd.GeoDataFrame(
#             points,
#             geometry = gpd.points_from_xy(x=points.x, y=points.y),
#         )

# print("Saving")
# points_gdf.to_file("/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/out_of_range_no_tree_points.shp",driver="ESRI Shapefile")

# ###################################################################

# print("Starting sampling of bioclimatic variables.")

# points_gdf = gpd.read_file("/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/out_of_range_no_tree_points.shp")

# yearly_temperature              = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_annual_mean_temp.tif"
# dry_quarter_temperature         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_driest_quarter.tif"
# wet_quarter_temperature         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_wettest_quarter.tif"
# cold_quarter_temperature        = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_coldest_quarter.tif"
# yearly_precipitation            = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_annual_precipitation.tif"
# wet_month_precipitation         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_wettest_month.tif"
# potential_evapotranspiration    = "/home/dibepa/git/global.agb.ml/data/training/raw/bioclimatic_data/et0_v3_yr.tif"
# temperature_seasonality         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_temperature_seasonality.tif"
# precipitation_seasonality       = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_seasonality.tif"

# bioclim_data = [
#     ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_isothermality.tif","iso"),
#     ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_max_temp_warmest_month.tif","mtwm"),
#     ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_diurnal_range.tif","mdr"),
#     ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_warmest_quarter.tif","mtwq"),
#     ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_min_temp_coldest_month.tif","mtcm"),
#     ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_coldest_quarter.tif","pcq"),
#     ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_driest_month.tif","pdm"),
#     ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_driest_quarter.tif","pdq"),
#     ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_warmest_quarter.tif","pwaq"),
#     ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_wettest_quarter.tif","pweq"),
#     ("/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_temperature_annual_range.tif","tar"),
#     (yearly_temperature,"yt"),
#     (dry_quarter_temperature,"tdq"),
#     (wet_quarter_temperature,"twq"),
#     (cold_quarter_temperature,"tcq"),
#     (yearly_precipitation,"yp"),
#     (wet_month_precipitation,"pwm"),
#     (potential_evapotranspiration,"pet"),
#     (temperature_seasonality,"ts"),
#     (precipitation_seasonality,"ps"),
# ]

# points_gdf.crs = "EPSG:4326"

# for path,col in bioclim_data:
#     points_gdf = add_feature_from_raster(points_gdf, col, path, "float")
#     print(points_gdf)

# points_gdf = add_feature_from_polygon_layer(points_gdf,"REALM","/home/dibepa/git/global.agb.ml/data/training/raw/biome_data/Ecoregions2017.shp","bgr")
# print(points_gdf)

# zero_abd = pd.DataFrame(points_gdf.drop(columns=['geometry','x','y']))
# zero_abd["abd"]=0.0

# print("Saving")
# zero_abd.to_csv("/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/out_of_range_no_tree_data.csv")

###################################################################

# Create new dataset

no_abd = pd.read_csv("/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/out_of_range_no_tree_data.csv")

cols = no_abd.columns.drop('Unnamed: 0')

abd = pd.read_csv("/home/dibepa/git/global.agb.ml/data/training/tmp_preprocessed/tallo_learning_preprocessed_onlybioclim.csv")

abd = abd[cols]

abd = abd.astype({'abd':'float32'})

abd = pd.concat([abd,no_abd],axis="rows")
abd = abd.reset_index(drop=True)
abd = abd.drop("Unnamed: 0",axis='columns')

abd.to_csv("/home/dibepa/git/global.agb.ml/data/training/final_onlybioclim_with_notree/tallo_learning_preprocessed_onlybioclim_notrees.csv")