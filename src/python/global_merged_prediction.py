from rioxarray.merge import merge_arrays,merge_datasets
import rioxarray as riox
from pathlib import Path
from natsort import natsorted
import rasterio
import pandas as pd
import numpy as np
import geopandas as gpd
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata,rasterize_image
import gc
import re

# dirABD = "/home/dibepa/git/global.agb.ml/data/predict/predicted.abd.onlybioclim/"
# files = natsorted(Path(dirABD).glob('ABD_lon*.csv'),key=str)

# dfABD = pd.DataFrame()
# for i,file in enumerate(files):
#     df = pd.read_csv(file)
#     dfABD = pd.concat([dfABD,df], axis = "rows")
#     print("Processed {} out of {} files.".format(i+1,len(files)))

# dfABD = dfABD.reset_index(drop=True)
# dfABD = dfABD.astype({'abd':'uint16'})
# dfABD = dfABD.astype({'x':'float32','y':'float32'})

# dfABD.to_csv("/home/dibepa/git/global.agb.ml/data/predict/predicted.abd.onlybioclim/ABD_global.csv",index=False)
# print(dfABD.abd.dtype)

# print(dfABD.abd.max())

# print(dfABD.abd.min())

# print(dfABD.abd.mean())

# print(dfABD.abd.median())



def create_raster(i,nchunks,dfABD):

    print("Processing {} out of {}".format(i+1,nchunks))
        
    print("Creating point layer.")
    pointsABD = gpd.GeoDataFrame(
            dfABD,
            geometry = gpd.points_from_xy(x=dfABD.x, y=dfABD.y),
            # dtype='uint16'
        )
    pointsABD=pointsABD.drop(["x","y"], axis="columns")
    
    print(pointsABD)
    # pointsABD.to_file("/home/dibepa/git/global.agb.ml/data/predict/predicted.abd.onlybioclim/ABD_global_{}.shp".format(i),format="ESRI Shapefile")
    # print("Saved.")

    res_deg = 0.00833
    res_sec = np.ceil(res_deg*3600)
    nodata = 9999

    print("Creating raster.")
    abd_raster = make_geocube(
                vector_data=pointsABD,
                measurements=["sd_abd"],
                resolution=(-res_deg, res_deg),
                # rasterize_function=partial(rasterize_points_griddata, filter_nan=False, fill = nodata, method = "linear"), 
                rasterize_function=rasterize_image, 
                fill = nodata
            )
    # abd_raster.abd.values.astype("uint16")
    print("Done.")

    # print("Saving raster.")
    # abd_raster.rio.to_raster('/home/dibepa/git/global.agb.ml/data/predict/predicted.abd.onlybioclim/ABD_global.tiff')
    # print("Done.")

    # print(abd_raster.dtype)

    return abd_raster

dirABD = "/home/dibepa/git/global.agb.ml/data/predict/predicted.abd.onlybioclim/with.notree.sd"
files = natsorted(Path(dirABD).glob('ABD_SD_lon*.csv'),key=str)

for i,file in enumerate(files): 
    df = pd.read_csv(file)
    print(df)
    df = df.astype({'sd_abd':'int16','x':'float32','y':'float32'})
    # df = df.astype({'abd':'uint16'})

    raster = create_raster(i,len(files),df)

    window = re.findall('ABD(.*)', file.stem) 
    print(window)

    raster.rio.to_raster("/home/dibepa/git/global.agb.ml/data/predict/predicted.abd.onlybioclim/with.notree.sd/tmp/ABD{}.tiff".format(window))  

dirABD = "/home/dibepa/git/global.agb.ml/data/predict/predicted.abd.onlybioclim/with.notree.sd/tmp"
files = natsorted(Path(dirABD).glob('ABD*.tiff'),key=str)

raster_list = [ riox.open_rasterio(file) for file in files ] 
print(len(raster_list))

print("Entering")
aux_list=[]
i=1
while len(raster_list)>1:
    print("iteration {}".format(i))    
    
    if not (len(raster_list)%2 == 0):
        aux_list.append(raster_list[-1])
        raster_list = raster_list[0:-1]

    raster_list = [ merge_arrays([tup[0],tup[1]]) for tup in zip(raster_list[0::2], raster_list[1::2]) ]
    print(len(raster_list))
    i+=1

print("Merging aux")
raster_aux = merge_arrays(aux_list)   

print("Merging final")
raster_final = merge_arrays([raster_aux,raster_list[0]])

raster_final.rio.to_raster("/home/dibepa/git/global.agb.ml/data/predict/predicted.abd.onlybioclim/with.notree.sd/ABD_SD_global_notree.tiff")        


# dfABD1 = pd.read_csv("/home/dibepa/git/global.agb.ml/data/predict/predicted.abd.onlybioclim/ABD_global.csv",dtype={'abd':'uint16','x':'float32','y':'float32'})

# nchunks = 128
# list_df = np.array_split(dfABD1,nchunks)

# del dfABD1
# gc.collect()

# n = 8
# list_chunks = [list_df[i * n:(i + 1) * n] for i in range((len(list_df) + n - 1) // n )] 

# del list_df
# gc.collect()

# final = []
# for j,l in enumerate(list_chunks[1:]):
#     raster_list = [ create_raster(i,nchunks,dfABD) for i,dfABD in enumerate(l) ]

#     while len(raster_list)>1:
#         raster_list = [ merge_datasets([tup[0],tup[1]]) for tup in zip(raster_list[0::2], raster_list[1::2]) ]

#     raster_list[0].rio.to_raster("/home/dibepa/git/global.agb.ml/data/predict/predicted.abd.onlybioclim/tmp/ABD_global_{}.tiff".format(j))        

#     final.append(raster_list[0])   

# while len(final)>1:
#     final = [merge_datasets ([tup[0],tup[1]]) for tup in zip(final[0::2], final[1::2]) ]    

# final[0].rio.to_raster("/home/dibepa/git/global.agb.ml/data/predict/predicted.abd.onlybioclim/ABD_global.tiff".format(j))       
    # mosaic = []

    # count = 1

    # files = natsorted(Path(rasterdir).glob('*.tiff'),key=str)

    # for i,file in enumerate(files):

    #     with rasterio.open(file) as src_dataset:
    #         meta = src_dataset.meta
    #         meta.update(
    #             {
    #                 "dtype": "float32"
    #             }
    #         )
    #         data = src_dataset.read()

    #         with rasterio.open("./global_predicted_map_2/float32/ABD{}.tiff".format(i), 'w', **meta) as dst_dataset:
    #             dst_dataset.write(data)


    # for i,file in enumerate(files):
    #     print(i)

    #     raster = riox.open_rasterio(file)

    #     if len(mosaic)>=2:
    #         print("Attempting to merge")
    #         merged_raster = merge_arrays(mosaic) 
    #         print("Success")
    #         merged_raster.rio.to_raster("./global_predicted_map_nofilter_tcov/final/ABD_merged{}_30s.tiff".format(count)) 
    #         count += 1
    #         mosaic = []

    #     mosaic.append(raster)
                    
    # print("Attempting to merge")
    # merged_raster = merge_arrays(mosaic) 
    # print("Success")
    # merged_raster.rio.to_raster("./global_predicted_map_nofilter_tcov/final/ABD_merged{}_30s.tiff".format(count))                 