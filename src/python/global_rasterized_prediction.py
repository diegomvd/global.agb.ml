from pathlib import Path
import geopandas as gpd
import pandas as pd
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata,rasterize_image
from functools import partial
import re
import numpy as np
from natsort import natsorted

datadir = "./predicted_data_nofilter_tcov/"
# datadir = "./rasterize_test/"

res_deg = 0.00833
res_sec = np.ceil(res_deg*3600)
nodata = -9999

global_points = pd.DataFrame(dtype='float32')

files = natsorted(Path(datadir).glob('*.csv'),key=str)

count = 1
for i,file in enumerate(files):

    print(i)

    window = re.findall("ABD(.*)",file.name)[0]

    data = pd.read_csv(file)

    points = gpd.GeoDataFrame(
        data,
        geometry = gpd.points_from_xy(x=data.x, y=data.y),
        dtype='float32'
    )
    points = points.set_crs('EPSG:4326')

    global_points = pd.concat([global_points,points],axis=0, ignore_index=True)

    if i>2*count-1:

        abd_raster = make_geocube(
            vector_data=global_points,
            measurements=["abd"],
            resolution=(-res_deg, res_deg),
            # rasterize_function=partial(rasterize_points_griddata, filter_nan=False, fill = nodata, method = "linear"), 
            rasterize_function=rasterize_image, 
            fill = nodata
        )

        abd_raster.rio.to_raster('./predicted_map/ABD_{}s_global_{}_nofilter_tcov.tiff'.format(res_sec,count))

        global_points = pd.DataFrame(dtype='float32')

        count +=1

abd_raster = make_geocube(
            vector_data=global_points,
            measurements=["abd"],
            resolution=(-res_deg, res_deg),
            # rasterize_function=partial(rasterize_points_griddata, filter_nan=False, fill = nodata, method = "linear"), 
            rasterize_function=rasterize_image, 
            fill = nodata
        )

abd_raster.rio.to_raster('./predicted_map/ABD_{}s_global_{}_nofilter_tcov.tiff'.format(res_sec,count))

# points.to_file('./rasterize_test/ABD_{}km{}.shp'.format(res_sec,window),driver="ESRI Shapefile")


# abd_raster.rio.to_raster('./rasterize_test/ABD_{}km{}.tiff'.format(res_sec,window))
