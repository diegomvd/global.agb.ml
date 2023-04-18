"""
This module includes functions used to group instances by spatial proximity and to aggregate feature values within groups of instances.
Two grouping methods are provided: 
    1- Intersecting buffers.
    2- Grid tiles.
Dataset instances are rows of a Point GeoDataFrame.

Created on Monday January 02 2023.
@author: Diego Bengochea Paz.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

######
# Functions specific to grouping by buffer intersection.
######

def check_instances_bounding_box(data,group,bboxl):
    xmin, ymin, xmax, ymax = data[data.buffid==group].total_bounds
    return ( (abs(xmax-xmin) < bboxl ) and ( abs(ymax-ymin) < bboxl ) )

def set_group_to_nan(data,group):
    data["buffid"] = np.where(
            (data["buffid"]==group),
            np.nan,
            data["buffid"]
        )  
    return data

def check_group_bounding_box(data,group,bboxl,stop):
    # This is true if bounding box of the group is smaller than bboxl.
    group_is_legal = check_instances_bounding_box(data,group,bboxl)
    
    if group_is_legal: 
        # If the new group is legal, then grouping process should continue.
        stop = False
    else:
        # If the group is not legal then the new group's id (bufferid) is set to Nan, and is ignored during the group update step.
        data = set_group_to_nan(data,group)

    return (data,stop)    

def check_bounding_box(data,bboxl):
    """
    Iterates through all the newly created groups, checks whether they all lay within a bounding box of given size and if not associates a Nan index to the group.   
    """

    # Changing to a meter based CRS. 
    data["geometry"] = data["geometry"].to_crs("EPSG:3857")

    groups = data.buffid.unique()

    for g in groups: 
        data, stop = check_group_bounding_box(data,g,bboxl,True)

    # Changing back to the degrees based CRS.
    data["geometry"] = data["geometry"].to_crs("EPSG:4326")    

    # Stop is True only if every group is illegal, meaning that further grouping can be avoided.
    return (data, stop)    

def create_buffers(data,radius):
    
    # Re-project coordinates in Web Mercator CRS to use distances in Meters. 
    data["geometry"] = data["geometry"].to_crs("EPSG:3857")
    # Create the buffer zones with the specified radius.
    data["geometry"] = data["geometry"].buffer(radius)

    return data

def intersect_buffers(data):
    # Dissolve intersecting buffers: this step looses all information on individual instances.
    data_groups = data.dissolve().explode(index_parts=True).reset_index(drop=True) 
    # Assign an Id to each group of instances.
    data_groups["buffid"] = range(data_groups.shape[0])
    # Convert back to EPSG:4326
    data_groups = data_groups.to_crs("EPSG:4326")

    return data_groups

def group_instances_with_intersecting_buffers(data,radius,index_col):
    """
    This function creates groups of instances when instances's buffer disks of specified radius intersect with each other.  
    :param data:
    :param radius:
    :return:
    """  

    # Select group Id and geometry columns and reference index.
    data_groups = data[[index_col,"gid","geometry"]]

    # Create the buffer zones with the specified radius.
    data_groups = create_buffers(data_groups,radius)

    # Intersect buffers.
    data_groups = intersect_buffers(data_groups)
    
    # Join the group information with the original datase by doing a spatial join between buffers and points.
    new_data = gpd.sjoin(data, data_groups).reset_index(drop=True).drop(["index_right",index_col+"_right","gid_right"],axis="columns").rename(columns = {index_col+"_left":index_col,"gid_left":"gid"})

    return new_data

def buffer_grouping_iteration(data,sample_thr,r,bboxl,index_col):
    """
    One iteration of the grouping by buffer process. This includes: 
    1- Filtering out groups with enough samples.
    2- Creating new groups by intersecting buffers.
    3- Check bounding box has legal size for every group.
    4- Update groups and dataset.
    """

    datab = filter_by_sample_number(data, sample_thr)

    # The twin dataset to perform buffering operation.
    datab = group_instances_with_intersecting_buffers(
        datab,
        r,
        index_col
    )

    datab, stop = check_bounding_box(datab,bboxl)

    # Update the dataset only if there's at least one new group that is legal.
    if not stop:
        datab = update_groups(datab,"buffid")
        data = update_dataset(data,datab,index_col)

    return (data, stop)


def buffer_grouping_loop(data,rmin,rmax,dr,bboxl,index_col,sample_thr,save_tmp,save_path):
    
    data_grouped = gpd.GeoDataFrame(data)

    # If there are not undersampled groups then grouping can stop.
    undersampled_groups = filter_by_sample_number(data_grouped,sample_thr).shape[0]
    r = rmin 
   
    print("In buffer grouping loop: ")

    while (undersampled_groups > 0) & (r<=rmax):
        print("r = " +str(r))

        data_grouped, stop = buffer_grouping_iteration(data_grouped,sample_thr,r,bboxl,index_col)
        
        if save_tmp:
            data_grouped.to_file(save_path+"_grouped_r_"+str(r)+".shp", driver="ESRI Shapefile")

        if stop :
            print("Exiting the grouping by buffer: no new legal groups.")
            break

        undersampled_groups = filter_by_sample_number(data_grouped,sample_thr).shape[0]
        r+=dr    
        
    if not save_tmp:    
        data_grouped.to_file(save_path+"_grouped_r_"+str(r)+".shp", driver="ESRI Shapefile")

    return (data_grouped, r)  


######
# Functions specific to grouping by lattice.
######


def create_square_lattice(data,l):
    # Create a square lattice with the bounds of the reduced dataset.
    # Adapted from:
    # https://gis.stackexchange.com/questions/269243/creating-polygon-grid-using-geopandas

    datab = gpd.GeoDataFrame(data)
    datab["geometry"] = datab["geometry"].to_crs("EPSG:3857")
    xmin, ymin, xmax, ymax = datab.total_bounds
    
    cols = list(np.arange(xmin, xmax + l, l))
    rows = list(np.arange(ymin, ymax + l, l))

    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x,y), (x+l, y), (x+l, y+l), (x, y+l)]))

    lattice = gpd.GeoDataFrame({'geometry':polygons}).set_crs("EPSG:3857")   
    lattice["tileid"] = range(lattice.shape[0])    
    lattice = lattice.to_crs("EPSG:4326")

    return lattice 


def group_instances_by_tile_membership(data,lattice):
    datab = data.to_crs("EPSG:4326")
    latticeb = lattice.to_crs("EPSG:4326")
    return gpd.sjoin(datab,latticeb).reset_index(drop=True)


def lattice_grouping_iteration(data,sample_thr,l,index_col):
    datab = filter_by_sample_number(data,sample_thr)
    lattice = create_square_lattice(datab,l)
    datab = group_instances_by_tile_membership(datab,lattice)
    datab = update_groups(datab,"tileid")
    data = update_dataset(data,datab,index_col)
    return data


def lattice_grouping_loop(data,lmin,lmax,dl,index_col,sample_thr,save_tmp,save_path):

    data_grouped = gpd.GeoDataFrame(data)
    undersampled_groups = filter_by_sample_number(data_grouped,sample_thr).shape[0]
    l = lmin

    print("In lattice grouping loop: ")

    while (undersampled_groups > 0) & (l<=lmax):
        print("l = " +str(l))

        data_grouped = lattice_grouping_iteration(data_grouped,sample_thr,l,index_col)

        if save_tmp:
            data_grouped.to_file(save_path+"_grouped_r_"+str(lmin)+"_l_"+str(l)+".shp", driver="ESRI Shapefile")

        undersampled_groups = filter_by_sample_number(data_grouped,sample_thr).shape[0]
        l+=dl

    if not save_tmp:    
        data_grouped.to_file(save_path+"_grouped_r_"+str(lmin)+"_l_"+str(l)+".shp", driver="ESRI Shapefile")

    return (data_grouped, l)


######
# General functions.
######

def initialize_groups(data):
    data["gid"] = range(data.shape[0])
    data["n_samples"] = data["gid"].map(data["gid"].value_counts())
    return data

def filter_by_sample_number(data,sample_thr):
    filtered_data = data.drop(data[data.n_samples > sample_thr].index)
    return filtered_data   

def filter_data_by_category(data, cat_col, cat_val):
    return data[data[cat_col]==cat_val]

def update_dataset(original,grouped,index_col):

    grouped = grouped.set_index(index_col)
    grouped = grouped[["gid","n_samples"]]
    original = original.join(grouped, on = index_col, rsuffix = "_new")
    original["gid"] = np.where(
        original["gid_new"].isna(),
        original["gid"],
        original["gid_new"]
        )
    original["n_samples"] = np.where(
        original["gid_new"].isna(), 
        original["n_samples"],
        original["n_samples_new"]
        )
    original = original.drop(["gid_new","n_samples_new"],axis="columns")
    return original

def update_group_id(data,new_group,new_group_col):
    # Get first appearing group ID.
    gid = data[data[new_group_col] == new_group].gid.head(1)
    # Set the same group ID for all the trees sharing an identical new group.
    data['gid'] = np.where(
        (data[new_group_col] == new_group), gid, data['gid']
    )
    return data

def update_number_of_samples(data):
    data["n_samples"] = data["gid"].map(data["gid"].value_counts()) 
    return data   

def update_groups(data,new_group_col):

    data = data.to_crs("EPSG:4326")

    groups = data[new_group_col].unique()
    groups_no_nan = groups[np.isfinite(groups)]

    for g in groups_no_nan:
        data = update_group_id(data,g,new_group_col)
    data = update_number_of_samples(data)

    data = data.drop(new_group_col,axis="columns")

    return data   

def depurate_data(data,ctg_col,index_col):
    data_depurated = data[[index_col,ctg_col,"geometry"]]
    return data_depurated

def reassemble_data(original,depurated,index_col):
    assembled = original.join(depurated, on = index_col, rsuffix = "_dep")
    return assembled

def group_by_category(data,ctg_col,rmin,rmax,dr,lmax,dl,bboxl,index_col,sample_thr,save_tmp,save_path): 
    
    data_dep = depurate_data(data,ctg_col,index_col)

    ctg_array = data[ctg_col].unique()

    data_ctg_list = []

    params_df = pd.DataFrame()

    data_init = initialize_groups(data_dep)

    for ctg in ctg_array: 

        data_grouped_ctg = filter_data_by_category(data_init,ctg_col,ctg)

        save_path_ctg = save_path + "_" + ctg_col + "_" + str(ctg)

        data_grouped_ctg, r, l = group_instances(data_grouped_ctg,rmin,rmax,dr,lmax,dl,bboxl,index_col,sample_thr,save_tmp,save_path_ctg)

        data_ctg_list.append(data_grouped_ctg)

        params_row = pd.DataFrame([{ctg_col:ctg, 'r':r, 'l':l }])
        params_df = pd.concat([params_df,params_row],axis=0, ignore_index=True)

    data_grouped = pd.concat(data_ctg_list,ignore_index=True,axis=0)
    data_grouped = data_grouped.set_index(index_col)
    data_grouped = data_grouped[["gid","n_samples"]]

    data_reassembled = reassemble_data(data,data_grouped,index_col) 

    return data_reassembled, params_df 


def group_instances(data,rmin,rmax,dr,lmax,dl,bboxl,index_col,sample_thr,save_tmp,save_path):
    data, r = buffer_grouping_loop(data,rmin,rmax,dr,bboxl,index_col,sample_thr,save_tmp,save_path)
    data, l = lattice_grouping_loop(data,r,lmax,dl,index_col,sample_thr,save_tmp,save_path)
    data = group_left_alones(data,l,sample_thr)
    return (data, r, l) 

def group_left_alones(data,l,sample_thr):
    lattice = create_square_lattice(data,l)
    datab = group_instances_by_tile_membership(data,lattice)
    tiles = datab.tileid.unique()
    for tile in tiles: 
        data_tile = datab[datab.tileid==tile]

        if data_tile.shape[0]==0:
            continue
    
        data_tile_oversampled = data_tile[data_tile.n_samples>sample_thr]
    
        if data_tile_oversampled.shape[0]==0:
            gid_low = data_tile.gid.head(1)
        else:    
            # Get group with lowest n_samples:
            gid_low = data_tile_oversampled[ data_tile_oversampled.n_samples == np.min(data_tile_oversampled.n_samples) ].gid.head(1)

        # print("Unique n_samples: ")
        # print(data_tile_oversampled.n_samples.unique() )
        # print("Min n_samples:")
        # print(np.min(data_tile_oversampled.n_samples))
        # print("gid_lowsss")
        # print(data_tile_oversampled[ data_tile_oversampled.n_samples == np.min(data_tile_oversampled.n_samples) ].gid)
        # print("gid_low")
        # print(gid_low)

        datab["gid"] = np.where(
            ((datab.n_samples<sample_thr) & (datab.tileid == tile)) ,
            gid_low,
            datab["gid"]
        )
    datab = datab.drop("tileid",axis="columns")
    datab = update_number_of_samples(datab)

    return datab

def aggregate_values_by_group(data,group_col,aggparams_list):
    """
    """ 
    aggfunc_dict = build_aggfunc_dict(aggparams_list)
    data_agg = data.groupby(group_col).agg(
        **aggfunc_dict
    )
    return data_agg

def add_to_aggfunc_dict(aggfunc_dict,col_names,aggfunc,suffix):
    if suffix:
        for col in col_names:
            aggfunc_dict[col+"_"+aggfunc]=pd.NamedAgg(column=col,aggfunc=aggfunc)
    else:
        for col in col_names:
            aggfunc_dict[col]=pd.NamedAgg(column=col,aggfunc=aggfunc)
    return aggfunc_dict    

def build_aggfunc_dict(aggparams_list):
    aggfunc_dict = {}
    for params in aggparams_list:
        cols = params[0]
        aggfunc = params[1]
        suffix = params[2]
        aggfunc_dict = add_to_aggfunc_dict(aggfunc_dict,cols,aggfunc,suffix)
    return aggfunc_dict  
        
 











