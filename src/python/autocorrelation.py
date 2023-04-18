"""
Module for testing and dealing with issues of spatial autocorrelation.
"""

import geopandas as gpd
import pandas as pd
import numpy as np 



def semivariance(values):
    print(values)
    return np.sum( np.array( [np.abs(row[1].get("val0") - row[1].get("val1"))**2 for row in values.iterrows()] ) ) / values.shape[0]

        
def pairwise_distance_matrix(gdf):
    return gdf.geometry.apply(lambda g: gdf.geometry.distance(g)) 


def distance_matrix_to_list(dmatrix):
    return dmatrix.stack().reset_index().rename({"level_0":"id_0","level_1":"id_1",0:"distance"},axis="columns")


def create_distance_bins(dlist,dcol,nbins, log_distance, equal_frequency):
    df = dlist[dlist[dcol]>0.0]
    if log_distance:
        if equal_frequency: # Not sure if this works...
            d = np.log(df[dcol])
            out, bins = pd.qcut(d,nbins,retbins=True,labels=False)
            bins = np.exp(bins)
        else:    
            min = np.log(np.min(df[dcol]) * 0.999) # to ensure the point is taken
            max = np.log(np.max(df[dcol]) * 1.001)
            bins = np.exp(np.linspace(min,max,nbins,endpoint=True))
    else:     
        if equal_frequency:
            out, bins = pd.qcut(d,nbins,retbins=True,labels=False)
        else:    
            min = np.min(df[dcol]) * 0.999 # to ensure the point is taken
            max = np.max(df[dcol]) * 1.001
            bins = np.linspace(min,max,nbins,endpoint=True)
    midpoints = 0.5 * (bins[:-1] + bins[1:])    
    return bins, midpoints

def get_bin_membership(distance, bins):
    index = -1
    while distance > bins[index+1]:
        index += 1
    return index

def group_distances_by_bin(dlist,dcol,bins):
    dlist["bin"] = dlist[dcol].apply(lambda d: get_bin_membership(d,bins))
    dlist = dlist[dlist.bin >= 0]
    return dlist 

def variogram_get_target_data(dlist, data, target_col, log):
    df = pd.DataFrame()
    for row in dlist.iterrows():
        id0 = int(row[1].get("id_0")) 
        id1 = int(row[1].get("id_1") )
        if log:
            # Exponential backtransform.
            val0 = np.exp(data[target_col].get(id0))
            val1 = np.exp(data[target_col].get(id1))
        else : 
            val0 = data[target_col].get(id0)
            val1 = data[target_col].get(id1)   
        
        new_row = pd.DataFrame( [{"val0":val0,"val1":val1}] )         
        df = pd.concat([df,new_row],axis=0, ignore_index=True)
       
    return df
        

def variogram(dlist, data, target_col, log, midpoints):
    df = pd.DataFrame()
    bins = np.unique(dlist.bin)
    for b in bins:
        print("Bin {}".format(b))
        dlb = dlist[dlist.bin == b]
        values = variogram_get_target_data(dlb,data,target_col,log)
        sv = semivariance(values)
        d = midpoints[b]
        new_row = pd.DataFrame( [{"bin":b,"sv":sv,"distance":d}] )         
        df = pd.concat([df,new_row],axis=0, ignore_index=True)
        print(df)
    return df

file = gpd.read_file("./temp_data/tallo_learning_preprocessed_spatial_under_1Mg.csv/tallo_learning_preprocessed_spatial_under_1Mg.shp").to_crs('+proj=cea')

# print(file.geometry.distance(file.geometry))

matrix = pairwise_distance_matrix(file)
dl = distance_matrix_to_list(matrix)
bins, midpoints = create_distance_bins(dl,"distance",20,log_distance = True)
dlb = group_distances_by_bin(dl,"distance",bins)
vg = variogram(dlb,file,"abd",log=True,midpoints=midpoints)

vg.to_csv("./temp_data/variogram_test_2.csv")



# print(matrix)