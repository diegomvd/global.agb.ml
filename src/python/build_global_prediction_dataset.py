import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd

# Raster datafiles.
yearly_temperature              = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_annual_mean_temp.tif"
dry_quarter_temperature         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_driest_quarter.tif"
wet_quarter_temperature         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_wettest_quarter.tif"
cold_quarter_temperature        = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_coldest_quarter.tif"
yearly_precipitation            = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_annual_precipitation.tif"
wet_month_precipitation         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_wettest_month.tif"
potential_evapotranspiration    = "/home/dibepa/git/global.agb.ml/data/training/raw/bioclimatic_data/et0_v3_yr.tif"
aridity_index                   = "/home/dibepa/git/global.agb.ml/data/training/raw/bioclimatic_data/ai_v3_yr.tif"
temperature_seasonality         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_temperature_seasonality.tif"
precipitation_seasonality       = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_seasonality.tif"
ndvi_summer                     = "/home/dibepa/git/global.agb.ml/data/training/raw/ndvi_data/ndvi_1999_2019_summer.tif"
ndvi_fall                       = "/home/dibepa/git/global.agb.ml/data/training/raw/ndvi_data/ndvi_1999_2019_fall.tif"
ndvi_winter                     = "/home/dibepa/git/global.agb.ml/data/training/raw/ndvi_data/ndvi_1999_2019_winter.tif"
ndvi_spring                     = "/home/dibepa/git/global.agb.ml/data/training/raw/ndvi_data/ndvi_1999_2019_spring.tif"
tree_cover                        = "/home/dibepa/Documents/GLADMeanTreeCover/glad_scrapping_builder/resampled_and_merged_30s_1km/resampled_30s_1km_global.tif" 

# Vector datafile.
biogeographic_realms = "/home/dibepa/git/global.agb.ml/data/training/raw/biome_data/Ecoregions2017.shp"

"""
Simplest approach to deal with rasters from different sources and different dimensions is to select a reference raster, vectorize it and then sample the rest of the rasters using the generated point layer to build a full dataframe with bioclimatic data and coordinates of the whole world at 5 arc-minutes resolution.
"""

with rasterio.open(yearly_temperature) as reference:
    
    # Define window dimensions to process by chunk: processing in 100 windows.
    ww = reference.width / 20
    wh = reference.height / 20

    # col_off_array = np.linspace(0,reference.width,10)
    # row_off_array = np.linspace(0,reference.height,10)

    col_off_array = np.arange(0,reference.width,ww)
    row_off_array = np.arange(0,reference.height,wh)

for i,col_off in enumerate(col_off_array):

    for j,row_off in enumerate(row_off_array):        

        if (i>-1 or (i==0 and j>=0)):

            print("Processing col: {} out of {} and row {} out of {}".format(i+1,len(col_off_array),j+1,len(row_off_array)))

            # Create window:
            window = rasterio.windows.Window(col_off,row_off,ww,wh)

            # Vectorize raster in that window:
            with rasterio.open(yearly_temperature) as reference:
                
                # Read on window.
                data = reference.read(
                    1,
                    window=window,
                    out_shape=(
                        reference.count,
                        int(window.height * 1.0),
                        int(window.width * 1.0)
                    ),
                    resampling = rasterio.enums.Resampling.bilinear
                )

                # Scale image transform
                # transform = reference.transform * reference.transform.scale(
                #     (reference.width / data.shape[-1]),
                #     (reference.height / data.shape[-2])
                # )

                # Extract raster data and coordinates in a numpy array.
                data = data[:,:]
                predictor = []
                for row,vec in enumerate(data):
                    for col,elem in enumerate(vec):
                        xy = reference.transform * (col+col_off,row+row_off)
                        predictor.append([elem, xy[0], xy[1]])
                predictor = np.array(predictor)
            
            # Create a DataFrame from the array.
            df = pd.DataFrame(predictor, columns=["yt","x","y"])

            # Remove NaN values from the dataframe.
            df = df.drop(
                df[ (df.yt < -280.0) ].index
            ).reset_index(drop=True)

            # Create a GeoDataFrame using the extracted coordinates. 
            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df.x, df.y)
            ).set_crs("EPSG:4326")

            coord_list = [(x,y) for x,y in zip(gdf['geometry'].x , gdf['geometry'].y)]

            # Open the rest of the rasters and sample them using the coordinates list: 
            raster_data = [
                (yearly_temperature,"yt"),
                (dry_quarter_temperature,"tdq"),
                (wet_quarter_temperature,"twq"),
                (cold_quarter_temperature,"tcq"),
                (yearly_precipitation,"yp"),
                (wet_month_precipitation,"pwm"),
                (potential_evapotranspiration,"pet"),
                (aridity_index,"ai"),
                (temperature_seasonality,"ts"),
                (precipitation_seasonality,"ps"),
                (ndvi_summer,"ndvism"),
                (ndvi_fall,"ndvif"),
                (ndvi_winter,"ndviw"),
                (ndvi_spring,"ndvisp"),
                (tree_cover,"tcov") 
            ]

            for path,col in raster_data:
                raster = rasterio.open(path)
                gdf[col] = [x for x in raster.sample(coord_list)]

            # Remove NaN values from the dataframe.
            # df = df.drop(
            #     df[ (df.tcov <= 0.0) ].index
            # ).reset_index(drop=True)    

            # Convert again to a DataFrame to explode the new values from a single-valued array to a float.
            df = pd.DataFrame(gdf)
            df = df.explode(
                ["yt","tdq","twq","tcq","yp","pwm","pet","ai","ts","ps","ndvism","ndvif","ndviw","ndvisp","tcov"]
            )

            gdf = gpd.GeoDataFrame(df).reset_index(drop=True)
            gdf["pet"] = gdf['pet'].astype(float)
            gdf["tdq"] = gdf["tdq"].astype(float)
            gdf["tcq"] = gdf['tcq'].astype(float)
            gdf["yt"] = gdf['yt'].astype(float)
            gdf["twq"] = gdf['twq'].astype(float)
            gdf["yp"] = gdf['yp'].astype(float)
            gdf["pwm"] = gdf['pwm'].astype(float)
            gdf["tcov"] = gdf['tcov'].astype(float)
            gdf["ai"] = gdf['ai'].astype(float)
            gdf["ts"] = gdf['ts'].astype(float)
            gdf["ps"] = gdf['ps'].astype(float)
            gdf["ndvism"] = gdf['ndvism'].astype(float)
            gdf["ndvif"] = gdf['ndvif'].astype(float)
            gdf["ndviw"] = gdf['ndviw'].astype(float)
            gdf["ndvisp"] = gdf['ndvisp'].astype(float)

            gdf = gdf.set_crs("EPSG:4326")

            
            (lon, lat) = reference.transform * (col_off,row_off)
            # gdf.to_file("./prediction_data/raster_predictors_global_data_lon_{}_lat_{}.shp".format(lon,lat), driver = "ESRI Shapefile")

            # Load BGR polygon layer
            bgr = gpd.read_file(biogeographic_realms)
            bgr = bgr[["REALM","geometry"]]
            bgr = bgr.rename({"REALM":"bgr"},axis="columns")
            bgr["geometry"] = bgr.geometry.to_crs("EPSG:4326")

            predictors = gpd.sjoin(gdf, bgr).reset_index(drop=True).drop("index_right",axis="columns")

            predictors_shp = predictors.drop(["x","y"],axis="columns")
            # predictors_shp.to_file("./prediction_data/predictors_global_data_lon_{}_lat_{}.shp".format(lon,lat), driver = "ESRI Shapefile")

            predictors_csv = pd.DataFrame(predictors.drop(["geometry"],axis="columns"))
            
            if len(predictors_csv.index) > 0:

                # # Remove areas with no-trees.
                # predictors_csv = predictors_csv.drop(
                #     predictors_csv[ (predictors_csv.tcov <= 0.0) ].index
                # ).reset_index(drop=True) 

                if len(predictors_csv.index) > 0:

                    predictors_csv.to_csv("./prediction_data_nofilter_tcov/predictors_global_data_lon_{}_lat_{}.csv".format(lon,lat),index=False)
