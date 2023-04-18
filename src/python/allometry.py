"""
This module contains the definition of Chave et al. 2014 AGB pan-tropical allometry that uses wood density, stem diameter and height, Jucker et al. 2017 AGB global allometry that uses height and crown radius differentiated by functional type (e.g. angiosperms and gymnosperms). The module includes data on wood density for a diverse set of regions from Chave et al. 2009, to be used with the Chave et al. 2014 allometry.

Created on Thursday February 17 2023.
@author: Diego Bengochea Paz.
"""

import numpy as np
import pandas as pd
import geopandas as gpd

def agb_exponents_jucker_2017(funcg):
    alpha = 0
    beta = 0
    if funcg == "Gymnosperm":        
        alpha = 0.093
        beta = -0.223

    return (alpha,beta)   

def agb_jucker_2017(alpha, beta, cr, h):
    return (0.016+alpha)*np.power(2*h*cr,2.013+beta)*np.exp(0.5*0.204**2)    

def hcr_jucker_2017(alpha,beta,d,hcr):
    k = np.exp(0.5*0.056**2)
    return 0.5/hcr*np.power( d/(k*np.exp(alpha)) , 1.0/beta)

def map_biome_to_forest_type(biome):
    forest_type = ""
    match biome:
        case 1.0:
            forest_type = "Tropical forests"
        case 2.0:
            forest_type = "Tropical forests"
        case 3.0:
            forest_type = "Tropical forests"        
        case 4.0:
            forest_type = "Temperate mixed forests"
        case 5.0:
            forest_type = "Temperate coniferous forests"
        case 6.0:
            forest_type = "Boreal forests"
        case 7.0:
            forest_type = "Woodlands and savannas"
        case 8.0:
            forest_type = "Woodlands and savannas"
        case 9.0:
            forest_type = "Woodlands and savannas"
        case 12.0:
            forest_type = "Woodlands and savannas"
        case 14.0: # mangroves
            forest_type = "Tropical forests"
        case _: # montane grasslands, deserts, tundra
            forest_type = "Not a forest"    
    return forest_type          

def d_allometry_exponents_jucker_2017(biogeographic_realm,forest_type, functional_group):
   
    alpha = 0.557
    beta = 0.809
   
    match biogeographic_realm:
        case "Afrotropic":
            if forest_type == "Tropical forests":
                    alpha = 0.428
                    beta = 0.821
            elif forest_type == "Woodlands and savannas":
                    alpha = 1.268
                    beta = 0.750
            else: 
                print("Afrotropic: wrong forest type.\n")         

        case "Australasia":
            match forest_type:
                case "Temperate mixed forests":
                    if functional_group == "Angiosperm":
                        alpha = 0.769
                        beta = 0.750
                    elif functional_group == "Gymnosperm":
                        alpha = 0.603
                        beta = 0.891
                    else: # functional group is Nan
                        alpha = 0.757
                        beta = 0.817  
                case "Woodlands and savannas":
                    alpha = 0.519
                    beta = 0.890
                case _: 
                    print("Australasia: wrong forest type. \n")  

        case "Indomalayan":
            alpha = 0.486
            beta = 0.802

        case "Nearctic":
            match forest_type:
                case "Boreal forests":
                    if functional_group == "Angiosperm":
                        alpha = 0.286
                        beta = 0.865
                    elif functional_group == "Gymnosperm":
                        alpha = 0.815
                        beta = 0.771
                    else:
                        alpha = 0.493
                        beta = 0.835    
                case "Temperate coniferous forests":
                    if functional_group == "Angiosperm":
                        alpha = 0.343
                        beta = 0.844
                    elif functional_group == "Gymnosperm":
                        alpha = 0.589
                        beta = 0.817
                    else:
                        alpha = 0.551
                        beta = 0.822          
                case "Temperate mixed forests":
                    if functional_group == "Angiosperm":
                        alpha = 0.367
                        beta = 0.887
                    elif functional_group == "Gymnosperm":
                        alpha = 0.861
                        beta = 0.755
                    else:
                        alpha = 0.381
                        beta = 0.884      
                case "Woodlands and savannas":
                    if functional_group == "Angiosperm":
                        alpha = 0.617
                        beta = 0.790
                    elif functional_group == "Gymnosperm":
                        alpha = 1.133
                        beta = 0.745
                    else:
                        alpha = 0.758
                        beta = 0.786          
        
        case "Neotropic":
            alpha = 0.591
            beta = 0.781

        case "Palearctic":
            match forest_type:
                case "Boreal forests":
                    if functional_group == "Angiosperm":
                        alpha = 0.467
                        beta = 0.839
                    elif functional_group == "Gymnosperm":
                        alpha = 1.430
                        beta = 0.649
                    else:
                        alpha = 1.395
                        beta = 0.646
                case "Temperate coniferous forests":
                    if functional_group == "Angiosperm":
                        alpha = 0.273
                        beta = 0.927
                    elif functional_group == "Gymnosperm":
                        alpha = 0.974
                        beta = 0.748
                    else:
                        alpha = 0.331
                        beta = 0.941
                case "Temperate mixed forests":
                    if functional_group == "Angiosperm":
                        alpha = 0.694
                        beta = 0.730
                    elif functional_group == "Gymnosperm":
                        alpha = 1.004
                        beta = 0.730
                    else:
                        alpha = 0.708
                        beta = 0.753 
                case "Tropical forests":
                    alpha = 0.540
                    beta = 0.791
                case "Woodlands and savannas":
                    if functional_group == "Angiosperm":
                        alpha = 0.910
                        beta = 0.738
                    elif functional_group == "Gymnosperm":
                        alpha = 1.373
                        beta = 0.717
                    else:
                        alpha = 0.819
                        beta = 0.785  
                case _: 
                    print("Palearctic: wrong forest type\n.")         

    return (alpha,beta)

def estimate_tree_height_jucker2017(bgr,biome,funcg,d,cr):
    forest_type = map_biome_to_forest_type(biome)
    alpha, beta = d_allometry_exponents_jucker_2017(bgr,forest_type,funcg)
    return hcr_jucker_2017(alpha,beta,d,cr)

def estimate_tree_crown_radius_jucker2017(bgr,biome,funcg,d,h):
    forest_type = map_biome_to_forest_type(biome)
    alpha, beta = d_allometry_exponents_jucker_2017(bgr,forest_type,funcg)
    return hcr_jucker_2017(alpha,beta,d,h)    

def estimate_tree_biomass_jucker2017(funcg,cr,h):
    alpha, beta = agb_exponents_jucker_2017(funcg)
    return agb_jucker_2017(alpha,beta,cr,h)    
                            

#################################################################
# Functions for Chave et al. 2014. Not used for the time being.


def agb_chave_2014(wood_density, diameter, height):
    return 0.0673*np.power(wood_density*diameter**2*height,0.976)

def wood_density_chave_2009(region):
    wd_dict = {
        "Africa (extratropical)" : 0.648,
        "Africa (tropical)" : 0.598,
        "Australia" : 0.725,
        "Australia/PNG (tropical)" : 0.636,
        "Central America (tropical)" : 0.560,
        "China" : 0.541,
        "Europe" : 0.525,
        "India" : 0.652,
        "Madagascar" : 0.662,
        "Mexico" : 0.676,
        "North America" : 0.540,
        "Oceania" : 0.604,
        "South America (extratropical)" : 0.715,
        "South America (tropical)" : 0.632,
        "South-East Asia" : 0.574,
        "South-East Asia (tropical)" : 0.574,
        "Total" : 0.613
    }
    return wd_dict(region)    

def biome_tropicality(biome):
    biome_id_dict = {
        1.0 : "tropical",
        2.0 : "tropical",
        3.0 : "tropical",
        4.0 : "extratropical",
        5.0 : "extratropical",
        6.0 : "extratropical",
        7.0 : "tropical",
        8.0 : "extratropical",
        9.0 : "extratropical",
        10.0: "extratropical", # montane grasslands and shrublands...
        11.0: "extratropicla", # tundra
        12.0: "extratropical", # mediterranean forests, woodlands, scrub
        13.0: "extratropical", # desert and xeric shrublands
        14.0: "tropical" #mangroves
    }
    return biome_id_dict(biome) 