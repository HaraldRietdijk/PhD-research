from steps.step_generic_code.general_functions import start_logging, get_run_id, complete_run
from steps.step_7.step_7_input_preparation import get_feature_groups ,get_dataframes, get_dataframe_per_cluster, split_dataframes
from steps.step_7.step_7_fitting import fit_clusters, make_prediction
from steps.step_7.step_7_storing_results import store_fitting_times

def do_step_7(app):
    # In this step models are calculated per cluster
    # Each dataframe contains all the data of the participants in a cluster
    # The trainingset are the aggregation of the trainingdata used in individual calculation
    def get_fitting_times_and_scores(row):
        print('Run id:'+str(row['run_id'])+', Feature_group:' + str(row['id']) + ', dataset:' + row['dataset'])
        cluster_dfs = get_dataframe_per_cluster(app, row, df_dataframes)
        variable_threshold = not(row['dataset'] == 'vfc')
        cluster_sets = split_dataframes(cluster_dfs, variable_threshold) #returns { cluster_id : {set_name : dataframe}}
        fitting_times_all_clusters = fit_clusters(cluster_sets)
        store_fitting_times(app, fitting_times_all_clusters, row['id'], run_id)
        make_prediction(app, folder, cluster_dfs, fitting_times_all_clusters, row, run_id)
    
    folder="results/clustering"
    start_logging(folder)
    run_id = get_run_id(app, 'Fitting on minimum Clusters', 'min_run', 7, 'vfc')

    print('Getting dataframes')
    df_dataframes = get_dataframes(app)
    print('Getting feature groups')
    feature_groups = get_feature_groups(app, run_id)
    print("Step 7: Fitting models")
    feature_groups['fitting_times'] = feature_groups.apply(get_fitting_times_and_scores, axis=1)
    complete_run(app, run_id)
