from auto_sklearn2 import AutoSklearnClassifier
from Orange.base import Learner, Model
from Orange.data import Table
import pandas as pd
import numpy as np

class OrangeAutoSklearnClassifier(Model):
    def __init__(self, learner: Learner, table: Table, max_runtime_secs: int = 60, seed: int|None = None):
        super().__init__(table.domain)
        self.name = "AutoSklearn"
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.columns = list(table.to_pandas_dfs()[0].columns)
        self.target_name = table.to_pandas_dfs()[1].columns[0]
        self.learn(learner, table)
        
    def learn(self, learner: Learner, data: Table):
        auto_sklearn = AutoSklearnClassifier(time_limit=self.max_runtime_secs, random_state=self.seed)
        auto_sklearn.fit(data.X, data.Y)
        self.model = auto_sklearn
    
    def predict(self, data: np.array) -> np.array:
        predictions = self.model.predict(data)
        return predictions
    
    def leaderboard(self) -> pd.DataFrame:
        leaderboard = []
        for model_name, score in self.model.get_models_performance().items():
            leaderboard.append({
                'model': model_name,
                'score': score
            })
        return pd.DataFrame(leaderboard)
        
class AutoSklearnLearner(Learner):
    name = 'AutoSklearn'
    __returns__ = OrangeAutoSklearnClassifier

    def __init__(self, max_runtime_secs=60, seed: int|None = None):
        super().__init__()
        self.max_runtime_secs = max_runtime_secs
        self.random_seed = seed

    def fit_storage(self, table: Table):
        return OrangeAutoSklearnClassifier(self, table, max_runtime_secs=self.max_runtime_secs, seed=self.random_seed)

if __name__ == "__main__":
    learner = AutoSklearnLearner()
    from Orange.data import Table
    data = Table('iris')
    classifier = learner.fit_storage(data)
    leaderboard = classifier.leaderboard()
    print(leaderboard)