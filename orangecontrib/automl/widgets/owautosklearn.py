from orangecanvas.localization import Translator  # pylint: disable=wrong-import-order
_tr = Translator("orangecontrib.automl", "biolab.si", "Orange")
del Translator


from AnyQt.QtCore import Qt
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from orangecontrib.automl.autosklearn import AutoSklearnLearner
debug = None

class OWAutoSklearn(OWBaseLearner):
    name = _tr.m[9, 'AutoSklearn']
    description = _tr.m[6, "Orange Widget for {}"].format(_tr.m[9, 'AutoSklearn'])
    icon = 'icons/autosklearn-logo.svg'
    priority = 80
    keywords = 'automl'
    LEARNER = AutoSklearnLearner
    max_runtime_secs = Setting(60)
    use_random_seed = Setting(False)
    random_seed = Setting(0)

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, _tr.m[2, "Settings"])

        self.random_seed_spin = gui.spin(box, self, 'random_seed', 0, 2 ** 31 - 1, controlWidth=80, label=_tr.m[0, 'Fixed seed for random generator'], alignment=Qt.AlignRight, callback=self.settings_changed, checked='use_random_seed', checkCallback=self.settings_changed)
        self.max_runtime_spin = gui.spin(box, self, 'max_runtime_secs', 0, 3600, controlWidth=80, label=_tr.m[3, 'Max runtime'], alignment=Qt.AlignRight, callback=self.settings_changed)
        gui.widgetLabel(box, label=_tr.m[1, 'Leaderboard'])
        self.leaderboard = gui.table(box, rows=10, columns=2)
        self.leaderboard.setHorizontalHeaderLabels([_tr.m[4,'Model'], _tr.m[5,'Score'] ])
        self.leaderboard.setColumnWidth(0, 225)
        self.leaderboard.setColumnWidth(1, 50)

    def create_learner(self):
        return self.LEARNER(max_runtime_secs=self.max_runtime_secs, seed=self.random_seed)

    def get_learner_parameters(self):
        return (('max_runtime_secs', self.max_runtime_secs), ('seed', self.random_seed))

    def update_model(self):
        super().update_model()
        if self.model is None:
            return
        leaderboard_df = self.model.leaderboard()
        if leaderboard_df is None:
            return
        leaderboard = sorted(leaderboard_df[['model', 'score']].values.tolist(), key=lambda x: x[1], reverse=True)
        for y in range(min(10, len(leaderboard))):
            for x in range(len(leaderboard[y])):
                if x == 0:
                    gui.tableItem(self.leaderboard, y, x, leaderboard[y][x])
                else:
                    gui.tableItem(self.leaderboard, y, x, '%0.3f' % leaderboard[y][x])

if __name__ == '__main__':
    WidgetPreview(OWAutoSklearn).run(Table('iris'))