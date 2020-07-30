###########################
## Author - Shuting Xing ##
## The following function is for showing the result of the model tuned with Optuna

import pandas as pd
import numpy as np


# import Optuna and OptKeras after Keras
import optuna 
print('Optuna', optuna.__version__)
from optuna.integration import TFKerasPruningCallback

from optkeras.optkeras import OptKeras
import optkeras
print('OptKeras', optkeras.__version__)

# (Optional) Disable messages from Optuna below WARN level.
optuna.logging.set_verbosity(optuna.logging.WARN)



def show_results(study):
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    # Show results from Optuna results

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    return study
