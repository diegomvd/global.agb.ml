import numpy as np
import pandas as pd

path = "./predictor_selection_2/best_predictors.csv"

selected_predictors = pd.read_csv("./predictor_selection_2/best_predictors.csv")

selected_predictors = selected_predictors[selected_predictors.predictors>3.0]

print(selected_predictors)

best_predictors_bag = pd.DataFrame()
for fold in np.unique(selected_predictors.fold):

    df1 = selected_predictors[selected_predictors.fold == fold]

    for n in np.unique(df1.predictors):

        df2 = df1[df1.predictors == n]
        # print(df2)

        max_samples = np.max(df2.samples)

        df2 = df2.drop( df2[df2.samples/max_samples < 0.5].index)
        # print(df2)

        row = df2[["predictors","combination"]]

        best_predictors_bag = pd.concat( [best_predictors_bag,row], axis = 0 , ignore_index=True )

best_predictors_bag = best_predictors_bag.drop_duplicates(subset='combination', keep="last").sort_values(by=['predictors']).reset_index(drop=True)


best_predictors_bag.to_csv("./training_data_XGBR/predictors.csv",index=True)