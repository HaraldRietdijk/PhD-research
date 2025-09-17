from sqlalchemy import func, VARCHAR, select, and_, text
from database.models.model import DBActivityModel
from database.models.view import view

dta = DBActivityModel.metadata.tables['hft_data_t']
ttt = DBActivityModel.metadata.tables['hft_treatment_t']
opr = DBActivityModel.metadata.tables['operation_data']
trh = DBActivityModel.metadata.tables['thresholds']
rst = DBActivityModel.metadata.tables['result_types']
rsm = DBActivityModel.metadata.tables['result_moments']
prs = DBActivityModel.metadata.tables['patient_results']
prw = DBActivityModel.metadata.tables['patient_job_recovery']
rwc = DBActivityModel.metadata.tables['rtw_classes']

sum_steps_ns = view(
    "sum_steps_ns",
    DBActivityModel.metadata,
    select(
        func.sum(dta.c.steps).label("daily_steps"),
        dta.c.treatment_id.label("treatment_id"),
        dta.c.year.label("year"),
        dta.c.week.label("week"),
        dta.c.weekday.label("weekday")
    )
    .select_from(dta)
    .group_by(dta.c.treatment_id, dta.c.year, dta.c.week, dta.c.weekday)
    .having(text('daily_steps>0'))
)

NS_SUM_STEPS_V = view(
    "ns_sum_steps_v",
    DBActivityModel.metadata,
    select(
        dta.c.id.label("id"),
        dta.c.treatment_id.label("treatment_id"),
        ttt.c.research_group.label("research_group"),
        opr.c.operation_date.label("operation_date"),
        dta.c.mdate.label("mdate"),
        dta.c.year.label("year"),
        dta.c.week.label("week"), 
        dta.c.weekday.label("weekday"), 
        dta.c.weekday.cast(VARCHAR).label("weekday_char"),
        dta.c.hour.label("hour"), 
        dta.c.steps.label("sum_steps"),
        func.hft_sum_steps_hour_f(dta.c.treatment_id,dta.c.year,dta.c.week,dta.c.weekday,dta.c.hour).label("sum_steps_hour"), 
        sum_steps_ns.c.daily_steps.label("daily_steps")
    )
    .select_from(sum_steps_ns.join(dta,and_(dta.c.treatment_id==sum_steps_ns.c.treatment_id,
                                         dta.c.year==sum_steps_ns.c.year,
                                         dta.c.week==sum_steps_ns.c.week,
                                         dta.c.weekday==sum_steps_ns.c.weekday))
                                         .join(ttt)
                                         .join(opr, dta.c.treatment_id==opr.c.treatment_id))
    .filter(and_(opr.c.operation_date!=None, ttt.c.research_group.in_((3,4))))
    .order_by(dta.c.treatment_id,ttt.c.research_group,dta.c.year,dta.c.week,dta.c.weekday,dta.c.hour),
)

class SumStepsNS(DBActivityModel):
    __table__ = NS_SUM_STEPS_V  

    __mapper_args__ = { 'primary_key':['id']}

    def __repr__(self):
        return f"SumSteps({self.id!r}, {self.treatment_id!r}, {self.research_group!r}, \
        {self.mdate!r}, {self.year!r}, {self.week!r}, {self.weekday!r}, \
        {self.weekday_char!r}, {self.hour!r}, {self.sum_steps!r}, {self.sum_steps_hour!r}, {self.daily_steps!r})"

PATIENT_RESULTS_VIEW = view(
    "patient_results_view",
    DBActivityModel.metadata,
    select(
        prs.c.id.label("id"),
        prs.c.treatment_id.label("treatment_id"),
        rst.c.code.label("result_type_code"),
        rst.c.description.label("result_type_description"),
        rsm.c.code.label("result_moment_code"),
        rsm.c.description.label("result_moment_description"),
        prs.c.int_value.label("int_value"), 
        prs.c.float_value.label("float_value"), 
        prs.c.string_value.label("string_value")
    )
    .select_from(prs.join(rst).join(rsm))
    .order_by(prs.c.treatment_id,rsm.c.code,rst.c.code),
)

class PatientResults(DBActivityModel):
    __table__ = PATIENT_RESULTS_VIEW  

    __mapper_args__ = { 'primary_key':['id']}

    def __repr__(self):
        return f"PatientResults({self.id!r}, {self.treatment_id!r}, {self.result_type_code!r}, \
        {self.result_type_description!r}, {self.result_moment_code!r}, {self.result_moment_description!r}, \
        {self.int_value!r}, {self.float_value!r}, {self.string_value!r}"

PATIENT_RTW_CLASS_VIEW = view(
    "patient_rtw_class_view",
    DBActivityModel.metadata,
    select(prw.c.treatment_id.label("treatment_id"),
           rwc.c.nr_of_classes.label("nr_of_classes"),
           rwc.c.class_id.label("class_id"),
           prw.c.definite_return_to_work_weeks.label("definite_return_to_work_weeks")
          )
    .select_from(prw.join(rwc,and_(prw.c.definite_return_to_work_weeks>=rwc.c.from_week, 
                                            prw.c.definite_return_to_work_weeks<rwc.c.to_week))
                )
)

class PatientRTWClass(DBActivityModel):
    __table__ = PATIENT_RTW_CLASS_VIEW  

    __mapper_args__ = { 'primary_key':['treatment_id']}

    def __repr__(self):
        return f"PatientRTWClass({self.treatment_id!r}, {self.class_id!r}, {self.definite_return_to_work_weeks!r}"

