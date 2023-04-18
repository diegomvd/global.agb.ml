import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous


def mean_absolute_error(ypred, y):
    return np.mean(np.abs(ypred-y))

def mean_signed_difference(ypred, y):
    return np.mean(ypred-y)

def mean_absolute_percentual_error(ypred, y):
    return 100 * np.mean(np.abs(ypred - y)/y)


# Create 5 data folds for nested cross-validation.
tallo_learn = pd.read_csv("/home/dibepa/git/global-above-ground-biomass-ml/revised_model/temp_data/tallo_learning_preprocessed.csv")

rng = np.random.default_rng()
id_array = tallo_learn.index.to_numpy()
rng.shuffle(id_array)

n_folds = 10
splits_id = np.array_split(id_array,n_folds)

for i in range(n_folds):
    test = tallo_learn.iloc[splits_id[i]]
    train = tallo_learn.drop(splits_id[i], axis = 0)
    #print(train)
    test.to_csv("/home/dibepa/git/global-above-ground-biomass-ml/revised_model/sklearn-gen/training_data_XGBR_round2/tallo_learning_preprocessed_outcvloop_test_fold_" + str(i+1) + ".csv",index=False)
    train.to_csv("/home/dibepa/git/global-above-ground-biomass-ml/revised_model/sklearn-gen/training_data_XGBR_round2/tallo_learning_preprocessed_outcvloop_train_fold_" + str(i+1) + ".csv",index=False)

# Three best sets of predictors from previous round.
pred1 = ["twq","yp","bgr1","bgr2","bgr3","ps","ts","ndvism","ndviw","wavai"]
pred2 = ["twq","tcq","pwm","ai","bgr1","bgr2","bgr3","ps","ts","ndvism","ndvif","ndviw"]
pred3 = ["yt","tdq","twq","tcq","yp","pwm","pet","tcov","bgr1","bgr2","bgr3","ts","ndvif"]
list_pred = [pred1,pred2,pred3]

df =pd.DataFrame()
for fold in [10,1]:
    
    data = pd.read_csv("/home/dibepa/git/global-above-ground-biomass-ml/revised_model/sklearn-gen/training_data_XGBR_round2/tallo_learning_preprocessed_outcvloop_train_fold_{}.csv".format(fold))
    test = pd.read_csv("/home/dibepa/git/global-above-ground-biomass-ml/revised_model/sklearn-gen/training_data_XGBR_round2/tallo_learning_preprocessed_outcvloop_test_fold_{}.csv".format(fold))

    for pred in list_pred:

        # Extract the target, weights and predictors.
        y = np.array(data["abd"])
        X = np.array(data[pred])

        # Initialize Extreme Gradient Boosting Regressor.
        xgbr = XGBRegressor(
            n_estimators=1000,
            eval_metric = "rmse",
            objective='reg:squarederror',
            )
        
        # Define nearest neighbor imputer with default parameters to impute nan NDVI values in boreal forests.
        imputer = KNNImputer()
        estimator = make_pipeline(imputer, xgbr)

        # Define space exploration for the genetic algorithm.
        param_grid = {'xgbregressor__learning_rate': Continuous(0.01, 0.6),
                    'xgbregressor__max_depth': Integer(3, 6),
                    'xgbregressor__min_child_weight': Continuous(1.0,10.0),
                    'xgbregressor__subsample': Continuous(0.6, 1.0),
                    'xgbregressor__min_split_loss': Continuous(0.0,100.0),
                    'xgbregressor__max_delta_step': Continuous(0.0,10.0)
                    }
        
        # Set up 5 folds for inner cross-validation with 1 repeat.
        cv = KFold(n_splits=5, shuffle = True)

        # Cross-validation with evolutionary alogirthm.
        evolved_estimator = GASearchCV(estimator=estimator,
                                    cv=cv,
                                    scoring="neg_root_mean_squared_error",
                                    param_grid=param_grid,
                                    n_jobs=-1,
                                    verbose=True,
                                    population_size=10,
                                    generations=200
                                    )
        
        evolved_estimator.fit(X,y)

        # Getting best parameter set for this predictor set and this fold.
        best = evolved_estimator.best_params_
        best["predictors"] = pred
        best["fold"] = fold

        # Get evaluation metrics by predicting on the outer folds of the nested-CV.
        Xtest = np.array(test[pred])
        ytest = np.array(test["abd"])
        ypred = evolved_estimator.predict(Xtest)

        ypred_exp = np.exp(ypred)
        y_exp = np.exp(ytest)

        best["mae"] = mean_absolute_error(ypred_exp, y_exp)
        best["msd"] = mean_signed_difference(ypred_exp, y_exp)
        best["mape"] = mean_absolute_percentual_error(ypred_exp, y_exp)
        
        df  = pd.concat([df,pd.DataFrame([best])], axis = 0 , ignore_index=True )
        print(df)

        df.to_csv("./xgbr_tuned_round2_fold_{}_missing.csv".format(fold),index=False)
df.to_csv("./xgbr_tuned_round2_all.csv",index=False)    