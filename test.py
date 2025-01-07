import numpy as np
from numpy import random
#import fastbook
#fastbook.setup_book()
#from fastbook import *
from fastai.tabular.all import *
import pandas as pd
#import matplotlib.pyplot as plt
#from fastai.imports import *
#np.set_printoptions(linewidth=130)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import os
import xgboost as xgb
#from xgboost import plot_importance
from xgboost import XGBClassifier
import warnings
import gc
import pickle
from joblib import dump, load
import typing as t
import bentoml

# Load the model by setting the model tag
booster = bentoml.xgboost.load_model("mental_health_v1:55nonrglewgf3xkp")

path = Path('data/')
test_df = pd.read_csv(path/'test.csv',index_col='id')

train_df = pd.read_csv(path/'train.csv',index_col='id')

cont_names,cat_names = cont_cat_split(train_df, dep_var='Depression')
splits = RandomSplitter(valid_pct=0.2)(range_of(train_df))
to = TabularPandas(train_df, procs=[Categorify, FillMissing,Normalize],
#to = TabularPandas(train_df, procs=[Categorify,Normalize],
                   cat_names = cat_names,
                   cont_names = cont_names,
                   y_names='Depression',
                   y_block=CategoryBlock(),
                   splits=splits)
dls = to.dataloaders(bs=64)
test_dl = dls.test_dl(test_df)
res = tensor(booster.predict(test_dl.xs))
print(res)

