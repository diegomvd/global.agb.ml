import pandas as pd
import numpy as np
from pathlib import Path
import re
import geopandas as gpd
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata,rasterize_image
from rioxarray.merge import merge_arrays,merge_datasets
import rioxarray as riox
from natsort import natsorted

# train_df = pd.read_csv("/home/dibepa/git/global.agb.ml/data/training/tmp_preprocessed/tallo_learning_preprocessed_onlybioclim.csv")

# hp_file = pd.read_csv("/home/dibepa/git/global.agb.ml/data/training/predictor_selection_onlybioclim/best_predictors_hp_absolute.csv")

# predict_dir = "/home/dibepa/git/global.agb.ml/data/predict/predictor.data.onlybioclim/"

# l = [ s.split("-") for s in hp_file.combination.tolist() ]
# predictors = set([item for sublist in l for item in sublist])
# predictors.discard("bgr")

# train_df = train_df[ predictors ]

# range_df = train_df.agg(["min","max"])

# def out_range(x,min,max):
#     if x<min:
#         return x - min 
#     if x>max:
#         return x -max
#     else:
#         return np.nan

# for i,file in enumerate(Path(predict_dir).glob('*')):

#     window = re.findall("predictors_global_data(.*)",file.name)[0]
#     print(window)

#     filename = "/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/predictors_in_range{}".format(window)

#     if not Path(filename).is_file(): 

#         prediction_data = pd.read_csv(file)

#         prediction_data = prediction_data.drop(prediction_data[(prediction_data.bgr == "Oceania")|(prediction_data.bgr == "Antarctica")].index)

#         if len(prediction_data.index)>0:
    
#             coords = prediction_data[["x","y"]]
#             X = prediction_data.drop(["x","y"],axis="columns")
#             X = X[ list(predictors) ]

#             X.replace([np.inf, -np.inf], np.nan, inplace=True)

#             for col in range_df.columns :
#                 X[col] = X[col].apply( lambda x: out_range(x,range_df[col]["min"],range_df[col]["max"]) )

#             X["in_range"] = X.isna().sum(axis=1)/12*100
#             X = X["in_range"]
#             df = pd.concat([X,coords], axis = "columns")
#             df.to_csv(filename, index=False)



#########################################################


# def create_raster(i,nchunks,df):

#     print("Processing {} out of {}".format(i+1,nchunks))
   
#     df = df.astype({'in_range':'uint16','x':'float32','y':'float32'})    
   
#     print("Creating point layer.")
#     points = gpd.GeoDataFrame(
#             df,
#             geometry = gpd.points_from_xy(x=df.x, y=df.y),
#             # dtype='uint16'
#         )
#     points=points.drop(["x","y"], axis="columns")
    
#     print(points)
    

#     res_deg = 0.00833
#     res_sec = np.ceil(res_deg*3600)
#     nodata = 9999

#     print("Creating raster.")
#     raster = make_geocube(
#                 vector_data=points,
#                 measurements=["in_range"],
#                 resolution=(-res_deg, res_deg),
#                 # rasterize_function=partial(rasterize_points_griddata, filter_nan=False, fill = nodata, method = "linear"), 
#                 rasterize_function=rasterize_image, 
#                 fill = nodata
#             )
#     print("Done.")

#     return raster

# dir = "/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/tmp"
# files = natsorted(Path(dir).glob('predictors_in_range*.csv'),key=str)

# for i,file in enumerate(files): 
#     df = pd.read_csv(file)
#     df = df.astype({'in_range':'uint16','x':'float32','y':'float32'})

#     raster = create_raster(i,len(files),df)

#     window = re.findall('predictors_in_range(.*)', file.stem) 
#     print(window)

#     raster.rio.to_raster("/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/tmp/predictors_in_range{}.tiff".format(window))  


# dir = "/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/tmp"
# files = natsorted(Path(dir).glob('predictors_in_range*.tiff'),key=str)

# dir = "/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim"
# files = natsorted(Path(dir).glob('predictor_range*.tiff'),key=str)

# raster_list = [ riox.open_rasterio(file) for file in files ] 

# print("Entering")
# aux_list=[]
# i=1
# while len(raster_list)>1:
#     print("iteration {}".format(i))    
    
#     if not (len(raster_list)%2 == 0):
#         aux_list.append(raster_list[-1])
#         raster_list = raster_list[0:-1]

#     raster_list = [ merge_arrays([tup[0],tup[1]]) for tup in zip(raster_list[0::2], raster_list[1::2]) ]
#     print(len(raster_list))

#     # if(len(raster_list)==8):
#     #     for j,r in enumerate(raster_list):
#     #         r.rio.to_raster("/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/predictor_range_global_{}.tiff".format(j))
#     #     break    

#     i+=1

# raster_list[0].rio.to_raster("/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/predictor_range_global_main.tiff")

# print("Merging aux")
# if len(aux_list)>0:
#     raster_aux = merge_arrays(aux_list)   
#     raster_aux.rio.to_raster("/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/predictor_range_global_aux.tiff")

main = riox.open_rasterio("/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/predictor_range_global_main.tiff")

aux = riox.open_rasterio("/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/predictor_range_global_aux.tiff")

raster_final = merge_arrays([main,aux])

# print("Merging final")
# raster_final = merge_arrays([raster_aux,raster_list[0]])

raster_final.rio.to_raster("/home/dibepa/git/global.agb.ml/data/predict/predictor.range.onlybioclim/predictor_range_global.tiff")