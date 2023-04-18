"""
Script to average multi-annual average of 1999-2019 dekadal data of Copernicus NDVI. All 36 dekades are averaged. Mathematically this does not represent an issue as all averages are over identical time-spans thus the average of the averages is the average of all. In result the script yields a layer with : 20-year average of annual NDVI value by season.  
"""

import requests
import numpy as np
import rasterio
import xarray as xr
import rioxarray
import sys
from rasterio.windows import Window
import os


manifests = [ 
    ("./ndvi_data/manifest_austral_summer.txt", "summer"),
    ("./ndvi_data/manifest_austral_fall.txt", "fall"),
    ("./ndvi_data/manifest_austral_winter.txt", "winter"),
    ("./ndvi_data/manifest_austral_spring.txt", "spring") 
    ]

def get_urls_from_manifest(manifest):
    with open(manifest) as file:
        urls = [line.rstrip() for line in file]
    return urls   

    
def process_array_by_chunks(X, f, n, scaling, offset):
    splits = np.array_split(X,n,axis=0)
    for i in range(len(splits)):
        splits[i] = f(splits[i],scaling,offset)
        # print(splits[i])
    return np.vstack(splits)    


def transformation(x, scaling, offset):
    """
    To convert from copernicus raster values to real values.
    """
    # print("transforming array")
    ret = 0
    if x>250:
        ret = np.nan
    else:
        ret = x*scaling + offset
    return ret


def read_raster_file(file, row_slice):
    """
    Reads the band of a raster file and returns it as numpy array with the real NDVI value.
    """

    # This treats netCDF files downloaded from copernicus to retrieve the mean NDVI between 1999 and 2019 and saves the result in GeoTiff format. 
    if ".nc" in file: 
        # Numbers from Copernicus.
        scaling = 0.004
        offset = -0.08

        # Convert raster values to NDVI values.
        f = np.vectorize(transformation)

        nc = xr.open_dataset(file)
        nc.rio.write_crs("epsg:4326", inplace=True)
        del nc["mean"].attrs['grid_mapping']
        nc["mean"].rio.to_raster('./ndvi_data/temp_raster.tif')
 
        with rasterio.open("./ndvi_data/temp_raster.tif") as src:
            print("Opened the raster ...")
            array = src.read(
                1,
                window = Window.from_slices((row_slice[0], row_slice[1]), (0, 40320))
            ).astype(float)
            print("Loaded it in array, started processing array of size (Mb) ...")
            print(sys.getsizeof(array)*np.power(10.0,-6.0))
            # Array is too big to be processed as a whole. I need to divide it in chunks.
            return process_array_by_chunks(array,f,100,scaling,offset)
    else:
        # Here files have already been treated at first level and the raster file to open is already an average of 3 layers in GeoTiff format. 
        with rasterio.open(file) as src:
            print("Opened the raster ...")
            array = src.read(
                1,
                window = Window.from_slices((row_slice[0], row_slice[1]), (0, 40320))
            ).astype(float)
            print("Loaded it in array, started processing array of size (Mb) ...")
            print(sys.getsizeof(array)*np.power(10.0,-6.0))
            return array

def average_rasters(file_list, output_path):
    """
    Collects a group of rasters as numpy arrays and performs the average. 
    :return: the path to the saved averaged raster.
    """

    if not os.path.exists(output_path):

        print("Attempting to average files :" + str(file_list) )
        
        average_list = []
        # nsplits = 20 # 20 vertical splits for array of 40320cols x 15680rows
        nsplits = 40 # 40 vertical splits for array of 40320cols x 15680rows
        for n in range(nsplits):
            
            # row_slice = (n*784,(n+1)*784)
            row_slice = (n*392,(n+1)*392)
            array_list = [read_raster_file(x,row_slice) for x in file_list]
        
            # Start averaging and re-splitting to prevent memory overflow.
            avg = np.mean(array_list,axis = 0)
            average_list.append(avg) 
            print(avg.shape)
            print("Size of average_list (Mb) ...")
            print(sys.getsizeof(average_list)*np.power(10.0,-6.0))
            # print("Data collected. Starting mean calculation ...")
            # for i in range(len(array_list)):
            #     array_list[i] = np.array_split(array_list[i],100,axis=0)
        
            #     # There are 3 arrays to average and 100 chunks.
            #     list = []
            #     for i in range(100):
            #         mean = np.mean([array_list[0][i], array_list[1][i], array_list[2][i] ], axis = 0)
            #         list.append(mean)
            # average_list.append(np.vstack(list)) 

        mean = np.vstack(average_list)

        print("Getting raster metadata ...")
        # Get metadata from one of the input files
        with rasterio.open("./ndvi_data/temp_raster.tif") as src:
            meta = src.meta

        meta.update(dtype=rasterio.float32)

        print("Writing averaged raster ...")
        # Write output file
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(mean.astype(rasterio.float32), 1)
            print("Success!")

    return output_path    

def download_raster_file(url, save_path):
    print("Downloading raster " + url + "\n")
    r = requests.get(url, auth=('dbengo', 'feramia'))
    print("Writing.")
    open(save_path, 'wb').write(r.content)
    print("Done.")
    return save_path


# Download all the files.
for file, season in manifests:
    urls = get_urls_from_manifest(file)
    for i,url in enumerate(urls):
        save_path = "./ndvi_data/ndvi_" + season + "_level1_" + str(i) + ".nc"
        if not os.path.exists(save_path):
            if not url == "":
                print("Downloading " + str(url))
                raster = download_raster_file(url,save_path)

for season in ["summer","fall","winter","spring"]:

    final_output_path = "./ndvi_data/ndvi_1999_2019_" + season + ".tif"

    if not os.path.exists(final_output_path):

        file_list = []
        for file in os.listdir("./ndvi_data/"):
            if "ndvi_" + season + "_level1" in file:
                file_list.append("./ndvi_data/"+file)

        # There are 9 layers per season. For memory reasons averaging is done 3-by-3. Averaging must always be done among sets of identical size. 9->3->1 final layer for the season. 
        level2 = []
        level1 = []

        counter = 0
        while len(level2) < 3: 

            if len(level1) < 3:
                level1.append(file_list[counter])
                counter += 1
            else:
                print(level1)
                output_path = "./ndvi_data/ndvi_average_" + season + "_level2_" + str(counter) + ".tif"
                avg = average_rasters(level1, output_path)
                level2.append(avg)    
                level1 = []

        avg = average_rasters(level2, final_output_path)



