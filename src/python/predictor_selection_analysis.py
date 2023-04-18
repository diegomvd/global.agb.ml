import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
from itertools import combinations

dir = "./predictor_selection_2/"
flist = []

# Each file is the last population of the GA used for predictor selection for each one of the 5 data folds.
for file in os.listdir(dir):
    if ".csv" in file:
        if dir + file not in flist:
            flist.append(dir+file)
        else:
            pass

# Treat every fold independently to embrace possible differences in the best predictors.
pselected = pd.DataFrame()

for file in flist:
    data = pd.read_csv(file)

    foldId = file[file.index("fold")+len("fold")]

    # #Plot the Pareto front.
    # sns.set_style("darkgrid", {"axes.facecolor": "0.925"})
    # sns.set_context(context="notebook",font_scale=1.1)
    # g = sns.catplot(data=data,kind="strip",x="objective$npredictors",y="objective$error",
    #             jitter=True, s=50, marker="o", linewidth=0.2, alpha=.9, color="coral", height = 7, #palette="deep",
    #             )
    # g.axes[0,0].set_xlabel("Number of predictors")
    # g.axes[0,0].set_ylabel("MSE")        
    # plt.savefig(dir+"accuracy-simplicity-tradeoff-fold-"+foldId+".svg", format = "svg")
    # plt.show()

    # Plot the occurrence of predictor combinations.
    feat_cols = ["yt","tdq","twq","tcq","yp","pwm","pet","ai","tcov","bgr","ps","ts","ndvism","ndvif","ndviw","ndvisp","wavai","w","evolution$samples"]
    
    # Filter out fits with MSE larger than 
    # data = data.drop(data[data["objective$error"]>1.18].index)

    npredictors = np.unique(data["objective$npredictors"])[1:] 

    for n in npredictors:
        datafilt = data[data["objective$npredictors"]==n]
        datafilt = datafilt[feat_cols]
        df = pd.DataFrame()
        for index, row in datafilt.iterrows():
            print(row)
            row = row[row>0]

            samples = row["evolution$samples"]
            row = row.drop("evolution$samples")

            predictors = [predictor for predictor in row.index]

            predictors = ('-').join(predictors)

            df_row = pd.DataFrame([{"Predictors": predictors, "Samples" : samples}])
            df = pd.concat([df,df_row],axis=0, ignore_index=True)

            pselected_row = pd.DataFrame([{"fold":foldId,"predictors":n,"combination": predictors, "samples" : samples}])
            pselected = pd.concat([pselected,pselected_row], axis = 0 , ignore_index=True)

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

pselected.to_csv(dir+"best_predictors.csv")
print(pselected)


