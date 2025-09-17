from sqlalchemy import func, VARCHAR, select, case, and_, text
from database.models.model import DBActivityModel
from database.models.view import view

mpr = DBActivityModel.metadata.tables['hft_model_parameters_t']
mdl = DBActivityModel.metadata.tables['hft_model_t']
prr = DBActivityModel.metadata.tables['hft_parameters_t']
pdn = DBActivityModel.metadata.tables['hft_prediction_t']
dta = DBActivityModel.metadata.tables['hft_data_t']
ttt = DBActivityModel.metadata.tables['hft_treatment_t']
mtc = DBActivityModel.metadata.tables['hft_metrics_t']

HFT_HYPERPARAMETERS_V = view(
    "hft_hyperparameters_v",
    DBActivityModel.metadata,
    select(
       prr.c.hft_algorithm_t_name.label("algorithm_name"), 
       prr.c.name.label("hyperparameter"), 
       mpr.c.value.label("value"), 
    )
    .select_from(mpr.join(mdl).join(prr))
    .order_by(mpr.c.treatment_id, prr.c.hft_algorithm_t_name, prr.c.name),
)

class HyperParameters(DBActivityModel):
    __table__ = HFT_HYPERPARAMETERS_V 

    __mapper_args__ = { 'primary_key':['algorithm_name', 'hyperparameter']}

    def __repr__(self):
        return f"HyperParameters({self.algorithm_name!r}, {self.hyperparameter!r}, {self.value!r})"

HFT_PREDICTION_CM_V = view(
    "hft_prediction_cm_v",
    DBActivityModel.metadata,
    select(
        dta.c.treatment_id.label("treatment_id"),
        dta.c.week.label("week"), 
        dta.c.weekday.label("weekday"), 
        dta.c.hour.label("hour"), 
        mdl.c.algorithm.label("algorithm"), 
        pdn.c.probability.label("probability"), 
        pdn.c.steps_cat.label("steps_cat"), 
        pdn.c.prediction_cat.label("prediction_cat"), 
        case((and_(pdn.c.steps_cat==1, pdn.c.prediction_cat==1), 'tp'),
              (and_(pdn.c.steps_cat==1, pdn.c.prediction_cat==0), 'fn'),
              (and_(pdn.c.steps_cat==0, pdn.c.prediction_cat==0), 'tn'),
              (and_(pdn.c.steps_cat==0, pdn.c.prediction_cat==1), 'fp'),
        ).label("confusion_matrix"),
    )
    .select_from(pdn.join(dta).join(mdl))
    .order_by(dta.c.week),
)

class PredictionCm(DBActivityModel):
    __table__ = HFT_PREDICTION_CM_V  

    __mapper_args__ = { 'primary_key':['treatment_id', 'week', 'weekday', 'hour', 'algorithm']}

    def __repr__(self):
        return f"PredictionCm({self.treatment_id!r}, {self.week!r}, {self.weekday!r}, \
        {self.hour!r}, {self.algorithm!r}, {self.probability!r}, \
        {self.steps_cat!r}, {self.prediction_cat!r}, {self.confusion_matrix!r})"


sum_steps = view(
    "sum_steps",
    DBActivityModel.metadata,
    select(
        func.sum(dta.c.steps).label("daily_steps"),
        dta.c.treatment_id.label("treatment_id"),
        dta.c.year.label("year"),
        dta.c.week.label("week"),
        dta.c.weekday.label("weekday")
    )
    .select_from(dta)
    .group_by(dta.c.treatment_id, dta.c.week, dta.c.weekday)
    .having(text('daily_steps>0'))
)

HFT_SUM_STEPS_V = view(
    "hft_sum_steps_v",
    DBActivityModel.metadata,
    select(
        dta.c.id.label("id"),
        dta.c.treatment_id.label("treatment_id"),
        ttt.c.research_group.label("research_group"),
        dta.c.year.label("year"),
        dta.c.week.label("week"), 
        dta.c.weekday.label("weekday"), 
        dta.c.weekday.cast(VARCHAR).label("weekday_char"),
        dta.c.hour.label("hour"), 
        dta.c.steps.label("sum_steps"),
        func.hft_sum_steps_hour_f(dta.c.treatment_id,dta.c.year,dta.c.week,dta.c.weekday,dta.c.hour).label("sum_steps_hour"), 
        sum_steps.c.daily_steps.label("daily_steps")
    )
    .select_from(sum_steps.join(dta,and_(dta.c.treatment_id==sum_steps.c.treatment_id,
                                         dta.c.week==sum_steps.c.week,
                                         dta.c.weekday==sum_steps.c.weekday))
                                         .join(ttt))
    .order_by(dta.c.treatment_id,ttt.c.research_group,dta.c.year,dta.c.week,dta.c.weekday,dta.c.hour),
)

class SumSteps(DBActivityModel):
    __table__ = HFT_SUM_STEPS_V  

    __mapper_args__ = { 'primary_key':['id']}

    def __repr__(self):
        return f"SumSteps({self.id!r}, {self.treatment_id!r}, {self.research_group!r}, \
        {self.year!r}, {self.week!r}, {self.weekday!r}, \
        {self.weekday_char!r}, {self.hour!r}, {self.sum_steps!r}, {self.sum_steps_hour!r}, {self.daily_steps!r})"

max_accuracy = view(
    "max_accuracy",
    DBActivityModel.metadata,
    select(
        ttt.c.treatment_id.label("treatment_id"), 
        mdl.c.algorithm.label("algorithm"), 
        mtc.c.f1_score.label("f1_score"),
        func.max(mtc.c.accuracy).label("max_accuracy"), 
        mtc.c.weekday.label("weekday")    
    )
    .select_from(mdl.join(mtc).join(ttt))
    .group_by(ttt.c.treatment_id, mdl.c.algorithm, mtc.c.f1_score, mtc.c.weekday)
)

HFT_TTT_AGM_ACCURACY_SCORE_V = view(
    "hft_ttt_agm_accuracy_score_v",
    DBActivityModel.metadata,
    select(
        mtc.c.id.label("metrics_id"),
        mtc.c.hft_treatment_id.label("hft_treatment_id"),
        mdl.c.algorithm.label("algorithm"),
        mtc.c.weekday.label("weekday"),
        mtc.c.accuracy.label("accuracy"),
        mtc.c.true_negative.label("true_negative"),
        mtc.c.true_positive.label("true_positive"),
        mtc.c.false_negative.label("false_negative"),
        mtc.c.false_positive.label("false_positive"),
        mtc.c.number_of_observations.label("number_of_observations"),
        mtc.c.threshold.label("threshold")
    )
    .select_from(mtc.join(mdl))
    .where(and_(mtc.c.accuracy==max_accuracy.c.max_accuracy,
                max_accuracy.c.treatment_id==mtc.c.hft_treatment_id))
    .group_by(mtc.c.id,
              mtc.c.hft_treatment_id,
              mdl.c.algorithm,
              mtc.c.weekday,
              mtc.c.accuracy,
              mtc.c.true_negative,
              mtc.c.true_positive,
              mtc.c.false_negative,
              mtc.c.false_positive,
              mtc.c.number_of_observations,
              mtc.c.threshold)
    .order_by(mtc.c.weekday),
)

class TreatmentAlgoAccScore(DBActivityModel):
    __table__ = HFT_TTT_AGM_ACCURACY_SCORE_V  

    __mapper_args__ = { 'primary_key':['metrics_id', 'hft_treatment_id', 'algorithm', 'weekday']}

    def __repr__(self):
        return f"SumSteps({self.metrics_id!r}, {self.hft_treatment_id!r}, {self.algorithm!r}, \
        {self.weekday!r}, {self.accuracy!r}, {self.true_negative!r}, \
        {self.true_positive!r}, {self.false_negative!r}, {self.false_positive!r}, {self.number_of_observations!r}, {self.threshold!r})"
