"""
This module contains functions to estimate Aboveground Biomass Density.
Created on Thursday February 17 2023.
@author: Diego Bengochea Paz.
"""

import numpy as np
import pandas as pd
import scipy
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, RandomizedSearchCV
import scipy.stats as stats


def optimal_group_agb_distribution(agb_samples, niter, bwmin, bwmax, kernel, in_splits, out_splits):
    
    bw_dist = {"bandwidth":  stats.uniform(bwmin,bwmax) }
    kde = KernelDensity(kernel=kernel)

    inner_cv = KFold(n_splits=in_splits, shuffle=True)
    outer_cv = KFold(n_splits=out_splits, shuffle=True)

    random_search = RandomizedSearchCV(
        estimator=kde, param_distributions=bw_dist, n_iter=niter, cv=inner_cv
    )

    best_params = random_search.fit(agb_samples).best_params_

    best_bw = best_params["bandwidth"]

    score = cross_val_score(
        random_search, X=agb_samples, cv = outer_cv
    ).mean()

    kde_tuned = KernelDensity(kernel=kernel, bandwidth = best_bw).fit(agb_samples)

    return (kde_tuned, score, best_bw)

def sample_group_agb_distribution(agb_distribution,tree_density):
    
    agb = 0.0
    for i in range(int(tree_density)):
        tree_agb = np.exp(agb_distribution.sample(1)[0])
        agb += tree_agb  

    return agb 

def estimate_group_biomass_density(agb_samples,tree_density,kdeparams,n_replicas):       
    
    agb_distribution, score, bw = optimal_group_agb_distribution(agb_samples,kdeparams["niter"],kdeparams["bwmin"],kdeparams["bwmax"],kdeparams["kernel"],kdeparams["in_splits"],kdeparams["out_splits"])
    
    abd_replicas = []
    for i in range(n_replicas):
        abd_replicas.append(sample_group_agb_distribution(agb_distribution,tree_density))

    median_abd = np.median(np.array(abd_replicas)) 
    mean_abd = np.mean(np.array(abd_replicas)) 
    std_abd = np.std(np.array(abd_replicas))   

    return (median_abd, mean_abd, std_abd, score, bw)  


def estimate_biomass_density(data, agb_col, dens_col, kdeparams, n_replicas, save_tmp, save_path):

    nsplits = np.max(np.array([kdeparams["out_splits"],kdeparams["in_splits"]]))

    data_nonan = data[np.isfinite(data[dens_col])]
    print(data_nonan.shape[0]) 
    data_filt = data_nonan[data_nonan.n_samples>nsplits]
    print(data_filt.shape[0]) 
    print(data.shape[0])

    data_filt = data_filt[["gid",agb_col,dens_col]]

    groups = data_filt.gid.unique()

    abd = pd.DataFrame()

    for g in groups:
        
        datag = data_filt[data_filt.gid == g]

        td_mean = np.mean(datag[dens_col])
        td_std = np.std(datag[dens_col])

        agb_array = np.array(datag[agb_col]).reshape(-1,1)
        
        if agb_array.shape[0] >= nsplits:

            med_abd, mean_abd, std_abd, score, bw = estimate_group_biomass_density(agb_array,td_mean,kdeparams,n_replicas)

            row = {"gid":g,
                "td_mean": td_mean,
                "td_std": td_std,
                "med_abd": med_abd,
                "mean_abd": mean_abd,
                "std_abd": std_abd,
                "score": score,
                "bw" : bw
                }
        else:
            print("Strange behavior:")
            print(g)
            row = {"gid":g,
                "td_mean": td_mean,
                "td_std": td_std,
                "med_abd": np.nan,
                "mean_abd": np.nan,
                "std_abd": np.nan,
                "score": np.nan,
                "bw" : np.nan
                }

        
        abdrow = pd.DataFrame([row])
        print()
        print(abdrow)
        print()

        abd = pd.concat([abd,abdrow], axis=0, ignore_index=True)

        if save_tmp:
            abd.to_csv(save_path)

    return abd

