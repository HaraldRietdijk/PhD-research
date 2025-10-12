from sklearn.feature_selection import SelectKBest
from steps.step_10.step_10_general_functions import append_scores_for_features, init_scores, save_method_results
from steps.step_generic_code.general_functions import complete_run, get_run_id
from steps.step_generic_code.general_variables.general_variables_all_shap import FILTER_METHODS, CLASSIFIERS

def get_filter_method_features(filter_methods, dataframes, features):
    filter_features = {}
    for method in filter_methods:
        print('Getting filter features for ', method[0])
        method_features = []
        for k in range(1,len(features)):
                select_method = SelectKBest(method[1],k=k)
                method_features.append(select_method.fit(dataframes['X_train'],dataframes['Y_class_train']).get_feature_names_out(features))
        filter_features[method[0]] = method_features
    return filter_features

def get_scores_for_filter_method(method_features_selection, dataframes, features):
    scores = init_scores()
    for method_features in method_features_selection:
        for name, classifier, _, _ in CLASSIFIERS:
            print(len(method_features),name)
            scores[name] = append_scores_for_features(scores, name, classifier, dataframes, method_features)
    return scores

def do_filter_methods(app, dataframes, features):
    filter_methods = FILTER_METHODS
    filter_features = get_filter_method_features(filter_methods, dataframes, features)
    for i in range(30):
        print('Starting filter run: ',str(i+1))
        run_id = get_run_id(app,"Feature Selection filter", 'test', 10, 'NS')
        filter_methods_scores = {}
        for method in filter_methods:
            filter_methods_scores[method[0]] = get_scores_for_filter_method(filter_features[method[0]], dataframes, features)
        save_method_results(app, filter_methods_scores, run_id, 'filter')
        complete_run(app, run_id)

