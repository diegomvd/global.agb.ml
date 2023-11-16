import pickle
from xgboost import XGBRegressor
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

# custom transformer for sklearn pipeline
class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return np.array(X[self.cols])

    def fit(self, X, y=None):
        return self

pickled_model = "./training_data/abd_model.pkl"
with open(pickled_model, 'rb') as f:
    model = pickle.load(f)

    model.save_model('./training_data/abd_model.json')