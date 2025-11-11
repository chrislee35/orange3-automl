from autogluon.tabular import TabularDataset, TabularPredictor
from Orange.base import Learner, Model
from Orange.data import Table
import pandas as pd
import numpy as np

class AutoGluonClassifier(Model):
    def __init__(self, learner: Learner, table: Table, max_runtime_secs: int = 60, eval_metric='accuracy'):
        super().__init__(table.domain)
        self.name = "AutoGluon"
        self.max_runtime_secs = max_runtime_secs
        self.eval_metric = eval_metric
        self.columns = list(table.to_pandas_dfs()[0].columns)
        self.target_name = table.to_pandas_dfs()[1].columns[0]
        self.learn(learner, table)
        
    def learn(self, learner: Learner, data: Table):
        x = list(data.to_pandas_dfs()[0].columns)
        train = TabularDataset(pd.concat([data.to_pandas_dfs()[0], data.to_pandas_dfs()[1]], axis=1))
        self.model = TabularPredictor(label=self.target_name, eval_metric=self.eval_metric).fit(train_data=train, time_limit=self.max_runtime_secs)
    
    def predict(self, data: np.array) -> np.array:
        X = pd.DataFrame(data, columns=self.columns)
        test = TabularDataset(X)
        predictions = self.model.predict(test)
        return predictions
    
    def leaderboard(self) -> pd.DataFrame:
        return self.model.leaderboard()
        
class AutoGluonLearner(Learner):
    name = 'AutoGluon'
    __returns__ = AutoGluonClassifier

    def __init__(self, max_runtime_secs=60, eval_metric='accuracy'):
        super().__init__()
        self.max_runtime_secs = max_runtime_secs
        self.eval_metric = eval_metric

    def fit_storage(self, table: Table):
        return AutoGluonClassifier(self, table, max_runtime_secs=self.max_runtime_secs, eval_metric=self.eval_metric)

if __name__ == "__main__":
    learner = AutoGluonLearner(eval_metric='accuracy')
    from Orange.data import Table
    data = Table('/home/chris/.local/share/Orange/3.39.0/datasets/core/breast-cancer.tab')
    classifier = learner.fit_storage(data)
    leaderboard = classifier.leaderboard()
    print(leaderboard)