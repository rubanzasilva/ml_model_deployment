import os
import numpy as np
import xgboost as xgb
import bentoml
from fastai.tabular.all import *
#from typing import Annotated  # Python 3.9 or above
#from typing_extensions import Annotated  # Older than 3.9
import pandas as pd
from bentoml.validators import DataframeSchema
import xgboost as xgb

@bentoml.service(
    resources={"cpu": "8"},
    traffic={"timeout": 10},
)

class MentalHealthClassifier:
    #retrieve the latest version of the model from the BentoML model store
    bento_model = bentoml.models.get("mental_health_v1:latest")
    #bento_model = BentoModel('mental_health_v1:q5kcqtf5ys3qoaav')


    def __init__(self):
        self.model = bentoml.xgboost.load_model(self.bento_model)

    def preprocess(self, data):
        path = Path('data/')
        train_df = pd.read_csv(path/'train.csv',index_col='id')
        test_df = pd.read_csv(path/'test.csv',index_col='id')
        cont_names,cat_names = cont_cat_split(train_df, dep_var='Depression')
        splits = RandomSplitter(valid_pct=0.2)(range_of(train_df))
        to = TabularPandas(train_df, procs=[Categorify, FillMissing,Normalize],
                           cat_names = cat_names,
                           cont_names = cont_names,
                           y_names='Depression',
                           y_block=CategoryBlock(),
                           splits=splits)
        dls = to.dataloaders(bs=64)
        test_dl = dls.test_dl(data)
        test_df_new = test_dl.xs
        return test_df_new
    

    @bentoml.api
    def predict(self, data:pd.DataFrame) -> np.ndarray:
        data = self.preprocess(data)
      # data = preprocess(data)

        prediction = self.model.predict(data)
        #prediction = torch.tensor(prediction)

        return prediction
       #if prediction == 0:
        #   status = "No Depression"
       #elif prediction == 1:
        #   status = "Depression"
       #else:
        #   status = "Error"
       #return status
       
        #Name = data.get("Name")
        #name_id = data.get("Name")
        
        
        #return {"prediction": prediction, "Name": Name}
        
        #return 

        #return self.model.predict(data)
