import time
import matplotlib.pyplot as plt
from sklearn import model_selection

from database.models.hft_tables import HFT_METRICS_GENERAL_MODELT, HFT_FITTING_TIME_T
from steps.step_generic_code.general_variables.general_variables import CLASSIFIERS_WITH_PARAMETERS
from steps.step_generic_code.general_functions import check_folder

def get_results(X_train_s, y_train_s,X_train, y_train,scoring):
    print("Step 2: Getting results: ", scoring)
    results = []
    names = []
    fitting_times_all_models = {}
    for name, classifier, normalized in CLASSIFIERS_WITH_PARAMETERS:
        start = time.time()
        kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=True)
        if not normalized:
            cv_results = model_selection.cross_val_score(classifier, X_train_s, y_train_s, cv=kfold, scoring=scoring)
        else:
            cv_results = model_selection.cross_val_score(classifier, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        end = time.time()
        fitting_times_all_models[name]=(end-start)
    return results, names, fitting_times_all_models

def plot_graph(results, names, title, plot_name, folder):
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    name = folder + '/times'
    check_folder(name)
    name = name + '/' + plot_name + '.png'
    plt.savefig(name, format='png')

def save_results(app, run_id, results, names, scoring):
    for index in range(len(results)):
        score = sum(results[index])/len(results[index])
        name = names[index]
        metric = HFT_METRICS_GENERAL_MODELT(algorithm = name,
                                            scoring = scoring,
                                            score = score,
                                            hft_run_id = run_id)
        app.session.add(metric)
    app.session.commit()

def save_times(app, run_id, times, random_seed):
    for name, time_in_secs in times.items():
        app.session.add(HFT_FITTING_TIME_T(hft_treatment_id = 0,
                                            algorithm = name, 
                                            fitting_time_sec = time_in_secs,
                                            random_seed = random_seed,
                                            hft_run_id = run_id))
    app.session.commit()