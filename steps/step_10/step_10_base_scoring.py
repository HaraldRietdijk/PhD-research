from steps.step_generic_code.general_functions import complete_run, get_run_id
from steps.step_generic_code.general_variables.general_variables_all_shap import CLASSIFIERS
from steps.step_10.step_10_general_functions import append_scores_for_features, init_scores, save_method_results


def do_base_scoring(app, dataframes, features):
    run_id = get_run_id(app,"Feature Selection filter", 'test', 10, 'NS')
    base_scoring = {}
    scores = init_scores()
    for i in range(30):
        print('Starting base run: ',str(i+1))
        for name, classifier, _, _ in CLASSIFIERS:
            print(str(i), name)
            scores[name] = append_scores_for_features(scores, name, classifier, dataframes, features)
    base_scoring['base'] = scores
    save_method_results(app, base_scoring, run_id, 'filter')
    complete_run(app, run_id)