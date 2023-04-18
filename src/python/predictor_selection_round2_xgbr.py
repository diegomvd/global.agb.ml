import numpy as np
import pandas as pd

cv_data = pd.read_csv("/home/dibepa/git/global-above-ground-biomass-ml/revised_model/xgbr_tuned_params_all.csv")

print(cv_data.groupby("predictors").mean("mape"))