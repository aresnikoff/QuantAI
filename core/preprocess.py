from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import pandas as pd
import numpy as np
import re

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('Preprocessing')
log.setLevel(logging.INFO)

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def _normalize(data):

  return preprocessing.scale(data, axis = 1)


def _qda(data):

  pass

def _lda(data):

  # maybe add priors
  return LDA(n_components = 2).transform(data)

def _pca(data):

  return PCA(n_components = data.shape[1]).fit_transform(data)

def _scale(state):

  state = pd.DataFrame(state)
  scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
  _state = scaler.fit_transform(state)
  return _state.flatten()

def _square(state):

  poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
  return poly.transform(state)


def gramian_angular_field(state, param_dict):

  # rescale data into 0-1
  state = state.apply(_scale, axis = 1)

  return state

def preprocess(state, param_dict):

  columns = state.columns

  state = gramian_angular_field(state, param_dict)

  price_col_names = [x for x in columns.tolist() if "price_" in x]
  price_cols = np.flip(natural_sort(price_col_names), axis = 0)

  state = pd.DataFrame(state, columns = columns)
  state = state[price_cols]
  return state

  # run_normalize = param_dict["normalize"]
  # run_pca = param_dict["pca"]
  # run_scale = param_dict["scale"]

  # columns = state.columns

  # if run_pca:

  #   x = _pca(state)

  #   state = pd.DataFrame(x, columns = columns)

  # if run_normalize:

  #   state = pd.DataFrame(_normalize(state), columns = columns)


  # # if run_lda:

  # #   state = _lda(state)

  # if run_scale:

  #   state = pd.DataFrame(_scale(state), columns = columns)

  # return state





