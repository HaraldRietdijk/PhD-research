from sqlalchemy.schema import DDL
from sqlalchemy import event
from database.models.model import DBActivityModel

def make_functions():
    drop_fn = "DROP FUNCTION IF EXISTS `hft_sum_steps_hour_f`"
    event.listen(
        DBActivityModel.metadata,
        'before_create',
        DDL(drop_fn)
        )

    hft_sum_steps_hour_f = (
        "CREATE DEFINER=`root`@`localhost` FUNCTION `hft_sum_steps_hour_f`(p_treatment_id int,p_year int,p_week int,p_weekday int,p_hour int) RETURNS int \n"
        "DETERMINISTIC \n"
        "BEGIN \n"
        "declare sum_steps_hour int DEFAULT 0;\n"
        "select sum(steps) into sum_steps_hour\n"
        "from hft_data_t\n"
        "where treatment_id=p_treatment_id\n"
        "and year= p_year\n"
        "and week=p_week\n"
        "and weekday=p_weekday\n"
        "and hour>=7\n"
        "and hour<=p_hour;\n"
        "if isnull(sum_steps_hour) then set sum_steps_hour=0;\n"
        "end if;\n"
        "RETURN sum_steps_hour;\n"
        "END\n"
        )
    event.listen(
        DBActivityModel.metadata,
        'before_create',
        DDL(hft_sum_steps_hour_f)
        )

    drop_fn = "DROP FUNCTION IF EXISTS `hft_sum_steps_f`"
    event.listen(
        DBActivityModel.metadata,
        'before_create',
        DDL(drop_fn)
        )
    hft_sum_steps_f = (
        "CREATE DEFINER=`root`@`localhost` FUNCTION `hft_sum_steps_f`(p_treatment_id int,p_year int,p_week int,p_weekday int) RETURNS int \n"
        "DETERMINISTIC \n"
        "BEGIN \n"
        "declare sum_steps int DEFAULT 0;\n"
        "select sum(steps) into sum_steps\n"
        "from hft_data_t\n"
        "where treatment_id=p_treatment_id\n"
        "and year=p_year\n"
        "and week=p_week\n"
        "and weekday=p_weekday;\n"
        "if isnull(sum_steps) then set sum_steps=0;\n"
        "end if;\n"
        "RETURN sum_steps;\n"
        "END\n"
        )
    event.listen(
        DBActivityModel.metadata,
        'before_create',
        DDL(hft_sum_steps_f)
        )
