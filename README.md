# Orange3-AutoML

This leverages several AutoML providers in [Orange3](https://orangedatamining.com/).

Within the Add-ons installer, click on "Add more..." and type in orange3-automl  
Currently the dependencies require **7.4 GBs** of free disk space.

![Installation Screenshot](imgs/install_orange3_automl.png)

Here is an example workflow using the iris dataset.

![Example Orange3 Workflow using iris and the H2O Learner](imgs/example_workflow.png)

## Selection of AutoML Providers

### Currently Supported

| Provider | Version | Notes |
|---------:|:--------|:------|
| [h2o](https://github.com/h2oai/h2o-3) | 3.4.4  | Works well. |
| [autogluon](https://auto.gluon.ai) | 1.4.0 | There a [bug on xgboost 2.0.3 through 2.1.3](https://github.com/autogluon/autogluon/issues/5288), which doesn't work with scikit-learn 1.7.1 (autogluon requires scikit-learn between 1.4.0 and 1.8.0) |
| [auto-sklearn2](https://github.com/agnelvishal/auto_sklearn2) | 1.0.0 | Works well. |


### Unsupported

I have tried the following packages and deemed them unsuitable.

| Provider | Version | Reason |
|---------:|:--------|:-------|
| mlbox    | 0.8.5   | Last updated 2020, requires old numpy (won't compile) |
| smac   | 2.3.1 | Requires installing swig before pip install, Orange interface doesn't have that interface |
| auto-sklearn | 0.24.2 | Last updated 2022, Won't compile on python 3.12 |
| tpot2 | 0.1.9a0 | Installs, but ran into issues with server hangs. |
| Google CloudAutoML | x | Cloud-based, requires accounts, sends data to Google |

### Unevaluated

I have not researched the following yet.

| Provider | Version | Reason |
|---------:|:--------|:-------|
| ludwig | | |
| transmogrifai | |
| evalml | | |
| MLJAR | | |
