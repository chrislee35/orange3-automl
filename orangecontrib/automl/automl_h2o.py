import h2o
from h2o.automl import H2OAutoML
from Orange.base import Learner, Model
from Orange.data import Table
import pandas as pd
import numpy as np

class H2OAutoMLClassifier(Model):
    def __init__(self, learner: Learner, table: Table, max_runtime_secs: int = 60, seed: int = 13):
        super().__init__(table.domain)
        self.name = "H2O AutoML"
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.columns = list(table.to_pandas_dfs()[0].columns)
        self.target_name = table.to_pandas_dfs()[1].columns[0]
        self.learn(learner, table)
        
    def learn(self, learner: Learner, data: Table):
        x = list(data.to_pandas_dfs()[0].columns)
        train = h2o.H2OFrame(pd.concat([data.to_pandas_dfs()[0], data.to_pandas_dfs()[1]], axis=1))
        train[self.target_name] = train[self.target_name].asfactor()
        self.model = H2OAutoML(max_runtime_secs = self.max_runtime_secs, seed = self.seed)
        self.model.train(x = x, y = data.domain.class_var.name, training_frame = train)
    
    def predict(self, data: np.array) -> np.array:
        X = pd.DataFrame(data, columns=self.columns)
        test = h2o.H2OFrame(X)
        predictions = self.model.leader.predict(test)
        y = np.array(predictions.as_data_frame())[:, 0]
        return y
    
    def leaderboard(self) -> pd.DataFrame:
        return self.model.leaderboard.as_data_frame()
        
class H2OAutoMLLearner(Learner):
    name = 'H2O AutoML'
    __returns__ = H2OAutoMLClassifier

    def __init__(self, max_runtime_secs=60, seed=1):
        super().__init__()
        h2o.init()
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed

    def fit_storage(self, table: Table):
        return H2OAutoMLClassifier(self, table, self.max_runtime_secs, self.seed)

