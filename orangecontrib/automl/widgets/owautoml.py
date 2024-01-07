from AnyQt.QtCore import Qt
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg
from orangecontrib.automl.automl import H2OAutoMLLearner

class OWAutoML(OWBaseLearner):
    name = "H2O AutoML"
    description = "Runs H2O AutoML"
    icon = "icons/h2o-logo.svg"
    priority = 80
    keywords = "automl"

    LEARNER = H2OAutoMLLearner

    max_runtime_secs = Setting(60)
    use_random_seed = Setting(False)
    random_seed = Setting(0)

    DEFAULT_BASE_ESTIMATOR = "DeepLearning"

    class Error(OWBaseLearner.Error):
        not_enough_features = Msg("Insufficient number of attributes ({})")

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


    def create_learner(self):
        return self.LEARNER(
            max_runtime_secs=self.max_runtime_secs,
            seed=self.random_seed
        )

    def get_learner_parameters(self):
        return (("Model", self.models[
                    self.model_index].capitalize()))


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWAutoML).run(Table("iris"))
