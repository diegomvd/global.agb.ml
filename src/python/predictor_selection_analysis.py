import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
from itertools import combinations
from pathlib import Path
import re

dir = "/home/dibepa/.openmole/dibepa-bc3/webui/projects/BiomassDensityML/tuning_XGBR/bioclimatic_learning/random_kfolds/results_onlybioclim"
flist = []

for path in Path(dir).iterdir():
    if path.is_dir:
        fold = path.name.split("_")[-1]
        max_gen = 0
        for file in Path(path).glob('*.csv'):
            name = file.name
            generation = int(re.search('population(.+?).csv', name).group(1))
            if generation>max_gen:
                max_gen = generation
                gen_file = file
        flist.append((fold,gen_file))

print(flist)


# Treat every fold independently to embrace possible differences in the best predictors.
selected = pd.DataFrame()

for fold,file in flist:
    data = pd.read_csv(file)
    
    # #Plot the Pareto front.
    # sns.set_style("darkgrid", {"axes.facecolor": "0.925"})
    # sns.set_context(context="notebook",font_scale=1.1)
    # g = sns.catplot(data=data,kind="strip",x="objective$npredictors",y="objective$error",
    #             jitter=True, s=50, marker="o", linewidth=0.2, alpha=.9, color="coral", height = 7, #palette="deep",
    #             )
    # g.axes[0,0].set_xlabel("Number of predictors")
    # g.axes[0,0].set_ylabel("MSE")        
    # plt.savefig("/home/dibepa/git/global.agb.ml/data/training/predictor_selection_onlybioclim/accuracy-simplicity-tradeoff-fold-"+fold+".svg", format = "svg")
    # plt.show()

    # Plot the occurrence of predictor combinations.
    pred_cols = ["yt","tdq","twq","tcq","yp","pwm","pet","iso","bgr","ps","ts","mtwm","mdr","mtwq","mtcm","pcq","pdm","pdq","pwaq","pweq","tar"]

    param_cols = ["yt","tdq","twq","tcq","yp","pwm","pet","iso","bgr","ps","ts","mtwm","mdr","mtwq","mtcm","pcq","pdm","pdq","pwaq","pweq","tar","e","md","mcw","mds","g","subsample","objective$error","evolution$samples"]


    npredictors = np.unique(data["objective$npredictors"])[1:] 

    for n in npredictors:
        datafilt = data[data["objective$npredictors"]==n]
        datafilt = datafilt[param_cols]
        samples_df = pd.DataFrame()
        error_df = pd.DataFrame()

        for index, row in datafilt.iterrows():
            
            # Simplify samples name.
            samples = row["evolution$samples"]
            row = row.drop("evolution$samples")
            error = row["objective$error"]
            row = row.drop("objective$error")

            row_pred = row[pred_cols]
            # Remove non-selected predictors.
            row_pred = row_pred[row_pred>0]

            predictors = [predictor for predictor in row_pred.index]

            predictors = ('-').join(predictors)

            samples_row = pd.DataFrame([{"Predictors": predictors, "Samples" : samples}])
            samples_df = pd.concat([samples_df,samples_row],axis=0, ignore_index=True)

            error_row = pd.DataFrame([{"Predictors": predictors,"Error":error, "e": row.e, "md":row.md, "mcw":row.mcw, "mds":row.mds, "g": row.g, "subsample":row.subsample}])
            error_df = pd.concat([error_df,error_row], axis = 0 , ignore_index=True)

        samples_df= samples_df.groupby("Predictors").sum("Samples").reset_index()
        predictor_set = samples_df.loc[samples_df["Samples"].idxmax()]["Predictors"]
        error_df = error_df[error_df.Predictors == predictor_set]
        selected_hp = error_df.loc[error_df["Error"].idxmin()]
        print(selected_hp)
        error_val = selected_hp.Error

        selected_row = pd.DataFrame([{
            "fold":fold,
            "predictors":n,
            "error" : error_val,
            "combination": selected_hp.Predictors,
            "e" : selected_hp.e,
            "md" : selected_hp.md,
            "mcw" : selected_hp.mcw,
            "mds" : selected_hp.mds,
            "g" : selected_hp.g,
            "subsample" : selected_hp.subsample
        }])

        selected = pd.concat([selected,selected_row], axis = 0 , ignore_index=True)

        # datafilt = datafilt.rename(columns={
        #     "yt" : "YT",
        #     "tdq": "TDQ",
        #     "twq": "TWQ",
        #     "tcq": "TCQ",
        #     "yp": "YP",
        #     "pwm": "WMP",
        #     "pet": "PET",
        #     "ai": "AI",
        #     "tcov": "TCO",
        #     "bgr": "BGR",
        #     "ts" : "TS",
        #     "ps" : "PS",
        # })

        # sns.set_color_codes(palette='muted')
        # sns.set_style("darkgrid", {"axes.facecolor": "0.925"})
        # sns.set_context(context="notebook",font_scale=1.1)
        # g = sns.catplot(data=df,kind="bar",y="Predictors",x="Samples",
        #                 color = "coral", height = 7, aspect = 1., linewidth=0.2, alpha=.9, #palette="pastel",
        #             )
        # g.axes[0,0].set_ylabel("Predictor combination")
        # g.axes[0,0].set_xlabel("Samples")    
        # g.axes[0,0].set_xlabel("Samples")
        # g.axes[0,0].set_xscale("log")
        # g.axes[0,0].set_title("Selected combinations with " + str(int(n)) + " predictors")   
        # plt.subplots_adjust(top=0.95,bottom=0.1)

        # plt.savefig(dir+"high.accuracy/selected-combinations-predictors-"+str(int(n))+"-fold-"+foldId+".svg", format = "svg")
        # plt.show()

# selected.to_csv("/home/dibepa/git/global.agb.ml/data/training/predictor_selection_onlybioclim/best_predictors_hp.csv")

best = selected.loc[selected.groupby("fold").error.idxmin()].reset_index(drop=True)
best.to_csv("/home/dibepa/git/global.agb.ml/data/training/predictor_selection_onlybioclim/best_predictors_hp_absolute.csv",index=False)
print(best)

# print(selected)


