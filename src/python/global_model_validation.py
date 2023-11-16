import pandas as pd
import geopandas as gpd
from dataset_creation import add_feature_from_raster
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv("/home/dibepa/git/global.agb.ml/data/map_validation/FOS_Plots_v2019.04.10_ABD_validation.csv")

df = df[ ["Lat_cnt","Lon_cnt","AGB_local","AGB_Feldpausch","AGB_Chave"] ]

df["geometry"] = list(zip(df.Lat_cnt, df.Lon_cnt))

# df = df.drop(["Lat_cnt","Lon_cnt"],axis="columns")

df = df.assign(mean=df.mean(axis=1))

df = df[ ["geometry","mean","Lat_cnt","Lon_cnt"] ]

df = df.groupby("geometry").mean()

df = df.reset_index()

df = df[ ["mean","Lat_cnt","Lon_cnt"] ]

print(df)

gdf = gpd.GeoDataFrame(
        df,
        geometry= gpd.points_from_xy(x=df.Lon_cnt, y=df.Lat_cnt)
    )

print(gdf["geometry"])

abd_raster = "/home/dibepa/git/global.agb.ml/data/predict/predicted.abd.onlybioclim/ABD_global.tiff"

gdf2 = add_feature_from_raster(gdf, "predicted", abd_raster, "uint16")

gdf2["diff"] = (gdf2["predicted"]-gdf2["mean"])/(gdf2["mean"])*100

avg = gdf2["diff"].mean()

print(gdf2)
print(avg)

print(gdf2["diff"].max())
print(gdf2["diff"].min())
print(gdf2["diff"].median())

gdf2 = gdf2[gdf2["diff"] < 2000]

sns.histplot(data=gdf2,x="diff")
plt.show()
