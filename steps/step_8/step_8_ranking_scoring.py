from database.models.patient_data import RFE_RANKING_SCORING

from steps.step_generic_code.general_functions import check_folder
from steps.step_8.step_8_plot_scoring import get_scoring, get_models_from_optimization, get_models

def get_maximum_accuracy(scoring):
    max_accuracy = scoring['accuracy'].max()
    max_accuracys = scoring.loc[scoring['accuracy']==max_accuracy]
    nr_features = max_accuracys['nr_selected'].min() 
    return nr_features, max_accuracy

def get_ranking_score_for_model(app, model, set_name, nr_of_features=44, nr_of_classes=2):
    scoring = {}
    scoring[set_name]  = get_scoring(app, model[0], nr_of_classes, set_name)
    nr_features_max_score, max_accuracy = get_maximum_accuracy(scoring[set_name])
    last_accuracy =.5
    rank_scoring = 1
    for nr_selected, accuracy in zip(scoring[set_name]['nr_selected'], scoring[set_name]['accuracy']):
        rel_distance = (abs(nr_selected - nr_features_max_score)/nr_of_features)*(1/(1-(nr_features_max_score/nr_of_features)))
        # rel_distance = (2*abs(nr_selected - nr_features_max_score)/nr_of_features)*((nr_features_max_score/nr_of_features))
        # rel_distance = abs(nr_selected - nr_features_max_score)*((nr_of_features-nr_features_max_score)/nr_of_features)
        if nr_selected<=nr_features_max_score:
            if accuracy<last_accuracy:
                rank_scoring-=rel_distance*(last_accuracy-accuracy)/max_accuracy
        else:
            if accuracy>last_accuracy:
                rank_scoring+=rel_distance*(last_accuracy-accuracy)/max_accuracy
        last_accuracy = accuracy
    rfe_ranking = RFE_RANKING_SCORING(estimator = model[0], nr_classes = 2, 
                                        nr_selected_max = nr_features_max_score, accuracy_max = max_accuracy,
                                        test_or_train_data = set_name, ranking_score = rank_scoring)
    app.session.add(rfe_ranking)

def get_ranking_score_per_model_for_set(app, folder, set_name, nr_of_classes=2):
    if set_name == 'test':
        models = get_models(app, nr_of_classes)
    else:
        models = get_models_from_optimization(app, nr_of_classes)
    for model in models:
        get_ranking_score_for_model(app, model, set_name)
    app.session.commit()

def get_ranking_score_per_model_for_optimization(app, folder, nr_of_classes=2):
    models = get_models_from_optimization(app, nr_of_classes)
    for model in models:
        get_ranking_score_for_model(app, model, 'test')
    app.session.commit()

