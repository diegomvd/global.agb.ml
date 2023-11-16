import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from multiprocessing import Pool
from pathlib import Path
import gc


# Raster datafiles.
yearly_temperature              = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_annual_mean_temp.tif"
dry_quarter_temperature         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_driest_quarter.tif"
wet_quarter_temperature         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_wettest_quarter.tif"
cold_quarter_temperature        = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_coldest_quarter.tif"
yearly_precipitation            = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_annual_precipitation.tif"
wet_month_precipitation         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_wettest_month.tif"
potential_evapotranspiration    = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/potential_evapotranspiration_v3_yr_1970_200.tif"
temperature_seasonality         = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_temperature_seasonality.tif"
precipitation_seasonality       = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_seasonality.tif"

isothermality = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_isothermality.tif"
max_temp_warm_month = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_max_temp_warmest_month.tif"
diurnal_range = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_diurnal_range.tif"
temp_warm_quarter = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_mean_temp_warmest_quarter.tif"
min_temp_cold_month = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_min_temp_coldest_month.tif"
precip_cold_quarter = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_coldest_quarter.tif"
precip_dry_month = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_driest_month.tif"
precip_dry_quarter = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_driest_quarter.tif"
precip_warm_quarter = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_warmest_quarter.tif"
precip_wet_quarter = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_precipitation_wettest_quarter.tif"
temp_annual_range = "/home/dibepa/Documents/worldclim_bioclim/HighResolution/wc2.1_30s_1970_2000_temperature_annual_range.tif"

# Vector datafile.
biogeographic_realms = "/home/dibepa/Documents/Ecoregions/Ecoregions2017.shp"

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


    window_list = []
    coord_dict = {}
    for i,col_off in enumerate(col_off_array):

        for j,row_off in enumerate(row_off_array):        

            (lon, lat) = reference.transform * (col_off,row_off)
            coord_dict[(col_off,row_off)] = (lon,lat)

            print("Processing col: {} out of {} and row {} out of {}".format(i+1,len(col_off_array),j+1,len(row_off_array)))
            # Create window:
            window = rasterio.windows.Window(col_off,row_off,ww,wh)
            window_list.append((window, col_off, row_off))


local = True

def vectorize(window, col_off, row_off):
            
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
        (lon, lat) = coord_dict.get((col_off,row_off))

        if local:
            filename = "/home/dibepa/git/global.agb.ml/data/predict/predictor.data.onlybioclim/predictors_global_data_lon_{}_lat_{}.csv".format(lon,lat)
        else:
            filename = "/home/ubuntu/diego/biomass/predictor.data.onlybioclim/predictors_global_data_lon_{}_lat_{}.csv".format(lon,lat)
        
        if not Path(filename).is_file():

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
           
            del df
            del predictor
            gc.collect()

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
                (temperature_seasonality,"ts"),
                (precipitation_seasonality,"ps"),
                (isothermality,"iso"),
                (max_temp_warm_month,"mtwm"),
                (diurnal_range,"mdr"),
                (temp_warm_quarter,"mtwq"),
                (min_temp_cold_month,"mtcm"),
                (precip_cold_quarter,"pcq"),
                (precip_dry_month,"pdm"),
                (precip_dry_quarter,"pdq"),
                (precip_warm_quarter,"pwaq"),
                (precip_wet_quarter,"pweq"),
                (temp_annual_range,"tar")
            ]

            for path,col in raster_data:
                raster = rasterio.open(path)
                gdf[col] = [x for x in raster.sample(coord_list)]

            # Convert again to a DataFrame to explode the new values from a single-valued array to a float.
            df = pd.DataFrame(gdf)


            df = df.explode(
                ["yt","tdq","twq","tcq","yp","pwm","pet","ts","ps",
                "iso","mtwm","mdr", "mtwq", "mtcm", "pcq", "pdm", "pdq", "pwaq", "pweq", "tar"]
            )

            gdf = gpd.GeoDataFrame(df).reset_index(drop=True)
            gdf["pet"] = gdf['pet'].astype(float)
            gdf["tdq"] = gdf["tdq"].astype(float)
            gdf["tcq"] = gdf['tcq'].astype(float)
            gdf["yt"] = gdf['yt'].astype(float)
            gdf["twq"] = gdf['twq'].astype(float)
            gdf["yp"] = gdf['yp'].astype(float)
            gdf["pwm"] = gdf['pwm'].astype(float)
            gdf["ts"] = gdf['ts'].astype(float)
            gdf["ps"] = gdf['ps'].astype(float)
            gdf["iso"] = gdf['iso'].astype(float)
            gdf["mtwm"] = gdf['mtwm'].astype(float)
            gdf["mdr"] = gdf['mdr'].astype(float)
            gdf["mtwq"] = gdf['mtwq'].astype(float)
            gdf["mtcm"] = gdf['mtcm'].astype(float)
            gdf["pcq"] = gdf['pcq'].astype(float)
            gdf["pdm"] = gdf['pdm'].astype(float)
            gdf["pdq"] = gdf['pdq'].astype(float)
            gdf["pwaq"] = gdf['pwaq'].astype(float)
            gdf["pweq"] = gdf['pweq'].astype(float)
            gdf["tar"] = gdf['tar'].astype(float)
            

            gdf = gdf.set_crs("EPSG:4326")

            print("Processing window Lon:{} Lat:{}".format(lon,lat))

            # Load BGR polygon layer
            bgr = gpd.read_file(biogeographic_realms)
            bgr = bgr[["REALM","geometry"]]
            bgr = bgr.rename({"REALM":"bgr"},axis="columns")
            bgr["geometry"] = bgr.geometry.to_crs("EPSG:4326")

            predictors = gpd.sjoin(gdf, bgr).reset_index(drop=True).drop("index_right",axis="columns")

            del gdf
            del bgr
            gc.collect()

            predictors_csv = pd.DataFrame(predictors.drop(["geometry"],axis="columns"))
            
            if len(predictors_csv.index) > 0:
                ret = "Saved window Lon:{} Lat:{}".format(lon,lat)

            else:
                ret = "Discarded window Lon:{} Lat:{}".format(lon,lat)    
            
            print(ret)
            predictors_csv.to_csv(filename,index=False)
            
            del predictors_csv
            gc.collect()

            return ret    
        
        else: 
            ret = "Window ({},{}) is already computed.".format(lon,lat) 
            print(ret)
            return ret


proc = 5
with Pool(processes = proc) as pool:
   
    print("Starting pool of {} workers".format(proc))
    result = pool.starmap(vectorize,window_list,chunksize=10)
    print(result)

print("Processing finished.")    
