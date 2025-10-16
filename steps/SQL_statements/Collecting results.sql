-- comparing test and optim results for ADABoost

SELECT r.estimator, max(s.value) ma, max(max_score.ma) oma, max(max_score.ma)-max(s.value) as improvement FROM activity.rfe_results r
join rfe_scoring s on r.id=s.rfe_results_id
join (SELECT max(sms.value) ma, rms.estimator me FROM activity.rfe_scoring sms
join rfe_results rms on rms.id=sms.rfe_results_id
where sms.test_or_train_data='optim' and sms.scoring_type_id=5
group by rms.estimator) as max_score
where r.estimator like 'ada%' and r.estimator = max_score.me
and not r.estimator in ('ADA', 'ADAR')
and s.test_or_train_data = 'test'
and s.scoring_type_id = 5
group by r.estimator
order by oma desc;

-- Getting optimum number of features fro RFE results

SELECT min(r.nr_selected), r.estimator, avg(max_score.ma) FROM activity.rfe_results r
join rfe_scoring s on r.id = s.rfe_results_id
join rfe_results_features f on r.id=f.rfe_results_id
join (SELECT max(sms.value) ma, rms.estimator me FROM activity.rfe_scoring sms
join rfe_results rms on rms.id=sms.rfe_results_id
where sms.test_or_train_data='optim' and sms.scoring_type_id=5
group by rms.estimator) as max_score
where r.estimator = max_score.me and test_or_train_data = 'optim' and scoring_type_id=5 
and s.value = max_score.ma
group by r.estimator;

-- Getting maximum accuracy score

select max(s.value) mv, min(r.nr_selected) min_s, r.estimator from rfe_scoring s
join rfe_results r on r.id = s.rfe_results_id
where s.scoring_type_id=5 and s.test_or_train_data='test'  and s.value in
(SELECT max(value) ma FROM activity.rfe_scoring ins
join rfe_results inr on inr.id=ins.rfe_results_id
where ins.test_or_train_data = 'test' and ins.scoring_type_id=5
and r.estimator = inr.estimator
group by inr.estimator)
group by r.estimator;

-- Getting features for maximum accuracy

SELECT re.estimator, re.nr_selected, mv, rf.feature, rf.coefficient, rf.importance FROM activity.rfe_results re
join rfe_results_features rf on rf.rfe_results_id = re.id 
join (select max(s.value) mv, min(r.nr_selected) min_s, r.estimator s_est from rfe_scoring s
join rfe_results r on r.id = s.rfe_results_id
where s.scoring_type_id=5 and s.test_or_train_data='test'  and s.value in
(SELECT max(value) ma FROM activity.rfe_scoring ins
join rfe_results inr on inr.id=ins.rfe_results_id
where ins.test_or_train_data = 'test' and ins.scoring_type_id=5
and r.estimator = inr.estimator
group by inr.estimator)
group by r.estimator) as min_val
where re.estimator = s_est and re.nr_selected = min_s
and rf.ranking = 99
order by mv desc, re.estimator;

-- STEP 7
-- select min and max rand index
SELECT algorithm, min(rand_index), max(rand_index) FROM activity.cluster_metric_t cm
join cluster_features_group_t cfg on cm.hft_feature_group_id=cfg.id
where cfg.hft_run_id in (156, 158, 163, 164,197)
group by cfg.algorithm
order by cfg.algorithm

-- Selecting the selected clusters per run and number of clusters
SELECT run_id_origin, algorithm, count(*), min(nr_of_features), max(nr_of_features), rand_index  FROM activity.clustering_selected_t cs
join cluster_features_group_t cfg on cfg.id=cs.hft_feature_group
join cluster_metric_t cm on cm.hft_feature_group_id = cs.hft_feature_group 
where cs.hft_run_id in 
	(SELECT run_id FROM activity.hft_run_t
		where run_type = 'definite' and run_step=7 and run_completed=1 and data_set = 'vfc')
group by algorithm, rand_index, run_id_origin
order by run_id_origin, algorithm

-- Characteristic appearance in the selected clusters
SELECT left(algorithm, length(algorithm)-1) al, characteristic_name, count(*) cc FROM activity.clustering_selected_t cs
join cluster_features_group_t cfg on cfg.id = cs.hft_feature_group
join cluster_features_t cf on cf.hft_feature_group_id = cfg.id
join hft_data_characteristic_t dc on dc.id = cf.hft_parameters_t_id
where cs.hft_run_id in (191, 190, 193, 201)
group by al, dc.characteristic_name
order by al, cc desc

-- classification results per cluster
SELECT fgm.hft_feature_group, fgm.cluster_group, cfg.algorithm as clustering, mo.algorithm as classifier,  
sum(cg.cluster_size) total_cluster, avg(accuracy) as accuracy_cluster, 
avg(f1_score) as f1_cluster
FROM activity.feature_group_cluster_metrics_t fgm
join cluster_features_group_t cfg on fgm.hft_feature_group = cfg.id 
join clustering_selected_t cs on cs.hft_feature_group=cfg.id and fgm.hft_run_id=cs.hft_run_id
join hft_model_t mo on mo.id=fgm.hft_model_id
join cluster_generated_t cg on cg.hft_feature_group=cfg.id and cg.cluster_group=fgm.cluster_group
where cs.hft_run_id in 
	(SELECT run_id FROM activity.hft_run_t
		where run_type = 'definite' and run_step=7 and run_completed=1 and data_set = 'vfc')
group by fgm.hft_feature_group, fgm.cluster_group, cfg.algorithm, mo.algorithm
order by fgm.hft_feature_group, clustering, classifier, fgm.cluster_group

-- Classification results per cluster group
SELECT fgm.hft_feature_group, cfg.algorithm as clustering, mo.algorithm as classifier,  
sum(accuracy*cluster_size/43) as gewogen_acc, sum(f1_score*cluster_size/43) as gewogen_f1, 
max(accuracy), max(accuracy) - min(accuracy) as spread_accuracy, max(f1_score), max(f1_score) - min(f1_score) as spread_f1_score
FROM activity.feature_group_cluster_metrics_t fgm
join cluster_features_group_t cfg on fgm.hft_feature_group = cfg.id 
join clustering_selected_t cs on cs.hft_feature_group=cfg.id and fgm.hft_run_id=cs.hft_run_id
join hft_model_t mo on mo.id=fgm.hft_model_id
join cluster_generated_t cg on cg.hft_feature_group=cfg.id and cg.cluster_group=fgm.cluster_group
where cs.hft_run_id in 
	(SELECT run_id FROM activity.hft_run_t
		where run_type = 'definite' and run_step=7 and run_completed=1 and data_set = 'vfc')
group by fgm.hft_feature_group, cfg.algorithm, mo.algorithm
order by classifier desc, spread_accuracy, clustering, fgm.hft_feature_group

-- Getting the Feature selection runs with the highest accuracy and from those the highest f1-score.
select mrf.* 
from fs_method_results mrf
join (select mr.method_id as method_id, mr.model as model, max(mr.accuracy) as acc, max(mr.f1_score) as f1, min(mr.nr_features) as feat
	from fs_method_results mr
	left join fs_method_results mrm on mr.method_id = mrm.method_id and mr.model = mrm.model 
			and  (mr.accuracy, mr.f1_score,-mr.nr_features) < (mrm.accuracy, mrm.f1_score,-mr.nr_features)
	where mrm.accuracy is null
	group by mr.method_id, mr.model) as mrmax
    on mrf.method_id = mrmax.method_id and mrf.model = mrmax.model 
		and mrf.accuracy = mrmax.acc and mrf.f1_score = mrmax.f1
		and mrf.nr_features = mrmax.feat
;

-- Getting average accuracies per model per method
select mr.method_id, mr.model, mr.nr_features, mr.threshold, avg(mr.accuracy) as acc, avg(mr.f1_score) as f1, avg(mr.precision), avg(mr.recall)
from fs_method_results mr
group by mr.method_id, mr.model, mr.nr_features, mr.threshold
order by mr.method_id, mr.model, acc desc, f1 desc;

select maxr.method_id, maxr.model, rf.feature, abs(avg(rf.coefficient)) as score, count(*)
from fs_method_results_features rf
join (select mrf.* 
		from fs_method_results mrf
		join (select mr.method_id as method_id, mr.model as model, max(mr.accuracy) as acc, max(mr.f1_score) as f1, min(mr.nr_features) as feat
			from fs_method_results mr
			left join fs_method_results mrm on mr.method_id = mrm.method_id and mr.model = mrm.model 
								and  (mr.accuracy, mr.f1_score,-mr.nr_features) < (mrm.accuracy, mrm.f1_score,-mr.nr_features)
			where mrm.accuracy is null
			group by mr.method_id, mr.model) as mrmax
		on mrf.method_id = mrmax.method_id and mrf.model = mrmax.model 
		and mrf.accuracy = mrmax.acc and mrf.f1_score = mrmax.f1
		and mrf.nr_features = mrmax.feat) as maxr on rf.result_id = maxr.id
group by maxr.method_id, maxr.model, rf.feature
having score=0
order by maxr.method_id, maxr.model, abs(avg(rf.coefficient)) desc, rf.feature;

-- counting features use and not used
select sm.type, sm.name, mr.model, count(*) 
from fs_method_results_features rf
join fs_method_results mr on mr.id = rf.result_id
join fs_selection_method sm on sm.id = mr.method_id
where sm.type <>'base' -- and rf.coefficient=0
group by sm.type, sm.name, mr.model
order by sm.type, sm.name, mr.model;

-- Get standard deviation of result per model
select underroot.model, round(avg(underroot.accuracy),2) as accuracy, round(sqrt(sum(underroot.vari)),2) as standard_dev from
(select dif.accuracy, dif.model, power(dif.diference,2)/total as vari from
(select mr.*, mr.accuracy-average.acc as diference, total
			from activity.fs_method_results mr
			join (SELECT mr2.method_id, mr2.model, avg(mr2.accuracy) as acc, count(*) as total 
					FROM activity.fs_method_results mr2
					where mr2.method_id=7
					group by mr2.method_id, model
					order by mr2.method_id, mr2.model) as average
					on average.method_id = mr.method_id and average.model = mr.model) as dif
                    ) as underroot 
	group by model
    order by model