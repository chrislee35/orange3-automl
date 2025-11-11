from orangecanvas.localization import Translator  # pylint: disable=wrong-import-order
_tr = Translator("orangecontrib.automl", "biolab.si", "Orange")
del Translator

from AnyQt.QtCore import Qt
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from orangecontrib.automl.autogluon import AutoGluonLearner
debug = None

class OWAutoGluon(OWBaseLearner):
    name = _tr.m[8, 'AutoGluon']
    description = _tr.m[6, "Orange Widget for {}"].format(_tr.m[8, 'AutoGluon'])
    icon = 'icons/autogluon-logo.svg'
    priority = 80
    keywords = 'autogluon'
    LEARNER = AutoGluonLearner
    max_runtime_secs = Setting(60)
    eval_metric = Setting(0)
    want_main_area = True
    resizing_enabled = True

    EVAL_METRICS = ['accuracy', 'balanced_accuracy', 'log_loss', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
            'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_macro', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_macro', 'roc_auc_ovr_micro',
            'roc_auc_ovr_weighted', 'average_precision', 'precision', 'precision_macro', 'precision_micro', 'precision_weighted',
            'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'mcc', 'pac_score'] + \
            ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'mean_absolute_percentage_error', 'r2', 'symmetric_mean_absolute_percentage_error']

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, _tr.m[2, "Settings"])

        self.eval_metric_combo = gui.comboBox(box, self, "eval_metric", items=OWAutoGluon.EVAL_METRICS, callback=self.settings_changed, label=_tr.m[10, "Evaluation metric"])
        self.max_runtime_spin = gui.spin(box, self, 'max_runtime_secs', 0, 3600, controlWidth=80, label=_tr.m[3, 'Max runtime'], alignment=Qt.AlignRight, callback=self.settings_changed)
        self.controlArea.layout().setAlignment(Qt.AlignTop)
        
        box2 = gui.widgetBox(self.mainArea, _tr.m[1, 'Leaderboard'])
        gui.widgetLabel(box2, label=_tr.m[1, 'Leaderboard'])
        self.leaderboard = gui.table(box2, rows=10, columns=2)
        self.leaderboard.setHorizontalHeaderLabels([_tr.m[4,'Model'], _tr.m[5,'Score'] ])
        self.leaderboard.setColumnWidth(0, 225)
        self.leaderboard.setColumnWidth(1, 50)
        self.mainArea.layout().setAlignment(Qt.AlignTop)

    def create_learner(self):
        eval_metric_name = self.EVAL_METRICS[self.eval_metric]
        return self.LEARNER(max_runtime_secs=self.max_runtime_secs, eval_metric=eval_metric_name)

    def get_learner_parameters(self):
        eval_metric_name = self.EVAL_METRICS[self.eval_metric]
        return (('max_runtime_secs', self.max_runtime_secs), ('eval_metric', eval_metric_name))

    def update_model(self):
        super().update_model()
        if self.model is None:
            return
        leaderboard_df = self.model.leaderboard()
        if leaderboard_df is None:
            return
        leaderboard = sorted(leaderboard_df[['model', 'score_val']].values.tolist(), key=lambda x: x[1], reverse=True)
        for y in range(min(10, len(leaderboard))):
            for x in range(len(leaderboard[y])):
                if x == 0:
                    gui.tableItem(self.leaderboard, y, x, leaderboard[y][x])
                else:
                    gui.tableItem(self.leaderboard, y, x, '%0.3f' % leaderboard[y][x])

if __name__ == '__main__':
    WidgetPreview(OWAutoGluon).run(Table('iris'))