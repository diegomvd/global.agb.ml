import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

data = pd.read_csv("./training_results_3/predicted_vs_actual_bgr_2.csv")

#data["predicted"] = np.log(data["predicted"])
#data["actual"] = np.log(data["actual"])

data = data[ data["predictors"] == 8 ]

# sns.scatterplot(data,x="actual",y="predicted",hue="bgr")
# # sns.lineplot(x=data["predicted"],y=data["predicted"],color="r", linewidth = 3)
# plt.yscale("log")
# plt.xscale("log")
# plt.show()

data["sd"] = data["predicted"]-data["actual"]

# sns.scatterplot(data,x="actual",y="sd",hue="bgr")
# # sns.lineplot(x=data["predicted"],y=0.0,color="r", linewidth = 3)
# plt.yscale("log")
# plt.xscale("log")
# plt.show()

data["ae"] = np.abs(data["sd"])

# sns.scatterplot(data,x="actual",y="ae",hue="bgr")
# # sns.lineplot(x=data["predicted"],y=0.0,color="r", linewidth = 3)
# plt.yscale("log")
# plt.xscale("log")
# plt.show()

data["ape"] = 100*np.abs(data["sd"]/data["actual"])

sns.scatterplot(data,x="actual",y="ape",hue="ndviw")
# sns.lineplot(x=data["actual"],y=20,color="r", linewidth = 3)
# sns.lineplot(x=data["predicted"],y=0.0,color="r", linewidth = 3)
plt.yscale("log")
plt.xscale("log")
plt.show()

sns.scatterplot(data,x="actual",y="ape",hue="wavai")
# sns.lineplot(x=data["actual"],y=20,color="r", linewidth = 3)
# sns.lineplot(x=data["predicted"],y=0.0,color="r", linewidth = 3)
plt.yscale("log")
plt.xscale("log")
plt.show()



