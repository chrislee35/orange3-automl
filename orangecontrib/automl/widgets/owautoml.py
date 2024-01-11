from AnyQt.QtCore import Qt
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from orangecontrib.automl.automl import H2OAutoMLLearner

debug = None

class OWAutoML(OWBaseLearner):
    name = "H2O AutoML"
    description = "Runs H2O AutoML"
    icon = "icons/h2o-logo.svg"
    priority = 80
    keywords = "automl"

    LEARNER = H2OAutoMLLearner

    max_runtime_secs = Setting(2)
    use_random_seed = Setting(False)
    random_seed = Setting(0)
    
    def add_main_layout(self):
        # this is part of init, pylint: disable=attribute-defined-outside-init
        box = gui.widgetBox(self.controlArea, "Parameters")

        self.random_seed_spin = gui.spin(
            box, self, "random_seed", 0, 2 ** 31 - 1, controlWidth=80,
            label="Fixed seed for random generator:", alignment=Qt.AlignRight,
            callback=self.settings_changed, checked="use_random_seed",
            checkCallback=self.settings_changed)

        self.max_runtime_spin = gui.spin(
            box, self, "max_runtime_secs", 0, 3600, controlWidth=80,
            label="Max Runtime for AutoML:", alignment=Qt.AlignRight,
            callback=self.settings_changed)
        
        gui.widgetLabel(box, label="Leaderboard")
        
        self.leaderboard = gui.table(box, rows=10, columns=2)
        self.leaderboard.setHorizontalHeaderLabels(['Model', 'MPCE'])
        self.leaderboard.setColumnWidth(0, 225)
        self.leaderboard.setColumnWidth(1, 50)
        
    def create_learner(self):
        return self.LEARNER(
            max_runtime_secs=self.max_runtime_secs,
            seed=self.random_seed
        )

    def get_learner_parameters(self):
        return (("max_runtime_secs", self.max_runtime_secs), ("seed", self.random_seed))

    def update_model(self):
        super().update_model()
        if self.model is None:
            return
        
        leaderboard_df = self.model.leaderboard()
        if leaderboard_df is None:
            return
        
        #self.leaderboard.setHorizontalHeaderLabels(leaderboard_df.columns)
        leaderboard = sorted(leaderboard_df[ ["model_id", "mean_per_class_error"] ].values.tolist(), key=lambda x: x[1])
        for y in range(min(10, len(leaderboard))):
            for x in range(len(leaderboard[y])):
                if x == 0:
                    gui.tableItem(self.leaderboard, y, x, leaderboard[y][x])
                else:
                    gui.tableItem(self.leaderboard, y, x, "%0.3f" % leaderboard[y][x])

if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWAutoML).run(Table("iris"))
