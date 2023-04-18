"""
Script to generate indices for a Buffered Leave-One-Out cross-validation scheme. 
"""

import geopandas as gpd
import pandas as pd
import numpy as np

def get_observable_ranges(data, cols):
    return {col: (data[col].max(), data[col].min()) for col in cols}

def valid_training_domain(instance, observable_ranges):
   
   bool = np.array([ (instance[predictor]<observable_ranges[predictor][0] & instance[predictor]>observable_ranges[predictor][1]) for predictor in observable_ranges])

   return np.all(bool) 

def bloo_folds(data, cols, rlist, nfolds, rng):
    
    folds = pd.DataFrame()
    for radius in rlist:  

        available_ids = [i for i in range(data.shape[0])]    
        used_ids = []

        while len(used_ids)<nfolds:

            id = available_ids[rng.integers(len(available_ids))]
            instance = data.iloc[id]
            buffered_ids = data.buffer(radius)[id]
            filtered = data.drop(buffered_ids)
            obs_ranges = get_observable_ranges(filtered,cols)

            if valid_training_domain(instance,obs_ranges):
                buffered_ids = data.buffer(radius)[id]
                fold = pd.DataFrame([ {"test_id" : id, "discarded_id" : buffered_ids, "radius": radius} ])
                folds = pd.concat([folds,fold],axis=0, ignore_index=True)
                available_ids.remove(id)
                used_ids.append(id)

    return folds    

    
data  = gpd.read_file("")
predictors = []

data = data[predictors]

rmin = 200
rmax = 1000
dr = 100
rlist = np.linspace(rmin,rmax,dr)

quantitative_predictors = []

nfolds = 500

rng = np.random.default_rng()

folds = bloo_folds(data, quantitative_predictors, rlist, rng)

folds.to_csv("")



