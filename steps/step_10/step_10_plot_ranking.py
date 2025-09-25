import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import func, and_

from database.models.patient_data import RFE_RESULTS, RFE_RESULTS_FEATURES
from steps.step_generic_code.general_functions import check_folder

def plot_rankings(ranking, folder, estimator, nr_of_classes):
    plt.rcParams["font.family"] = "serif"
    x = np.arange(len(ranking))
    fig, ax1 = plt.subplots()
    ax1.set_xticks(x)
    ax1.set_xticklabels(ranking['feature'], rotation=45, size='x-large', ha='right', rotation_mode='anchor')
    ax1.set_xlabel("Feature")
    ax1.set_ylabel('Average ranking')
    ax1.plot(x,ranking['avg_ranking'],color='tab:red', label='avg_ranking')
    title = 'Average rankings per feature'
    if nr_of_classes>0:
        title += ' for' + str(nr_of_classes) + ' classes'
    plt.title(title)
    name = folder + '/rankings'
    check_folder(name)
    name = name + '/' + str(nr_of_classes) + 'rankings_' + estimator + '.png'
    plt.savefig(name, format='png')

def get_ranking_models(app, nr_of_classes):
    ranking_models_query = app.session.query(RFE_RESULTS.estimator)\
                                 .filter(RFE_RESULTS.nr_classes==nr_of_classes)\
                                 .group_by(RFE_RESULTS.estimator)\
                                 .all()
    ranking_models = pd.DataFrame([row for row in ranking_models_query],
                          columns=['estimator'])
    return ranking_models

def get_rankings(app, estimator, nr_of_classes):
    ranking_query = app.session.query(RFE_RESULTS_FEATURES.feature, func.max(RFE_RESULTS_FEATURES.ranking))\
                                 .join(RFE_RESULTS)\
                                 .filter(and_(RFE_RESULTS.estimator==estimator,
                                              RFE_RESULTS.nr_classes==nr_of_classes))\
                                 .group_by(RFE_RESULTS_FEATURES.feature)\
                                 .order_by(func.max(RFE_RESULTS_FEATURES.ranking))\
                                 .all()
    ranking = pd.DataFrame([row for row in ranking_query],
                          columns=['feature','avg_ranking'])
    return ranking

def plot_ranking_per_model(app, folder, nr_of_classes):
    ranking_models = get_ranking_models(app, nr_of_classes)
    for model in ranking_models['estimator']:
        ranking = get_rankings(app, model, nr_of_classes)
        plot_rankings(ranking, folder, model, nr_of_classes)
