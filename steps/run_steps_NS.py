from steps.read_raw_data.read_raw_data_NS import do_read_raw_data_NS
from steps.step_1.step_1_NS import do_step_1_NS
from steps.step_2.step_2_NS import do_step_2_NS
from steps.step_3.step_3_NS import do_step_3_NS
from steps.step_3.step_3_graph import do_step_3_graph
from steps.step_5.step_5 import do_step_5
from steps.step_6.step_6 import do_step_6
from steps.step_7.step_7 import do_step_7
from steps.step_7.step_7b import do_step_7b
from steps.step_8.step_8 import do_step_8
from steps.step_9.step_9 import do_step_9
from steps.step_10.step_10 import do_step_10

import sys

def do_run_NS(app):
    run_type = sys.argv[2]
    run_all_steps = (run_type=="all")
    if (run_type=="rawdata"):
        do_read_raw_data_NS(app)
    if (run_type=="step1") or run_all_steps:
        do_step_1_NS(app)
    if (run_type=="step2") or run_all_steps:
        do_step_2_NS(app)
    if (run_type=="step3") or run_all_steps:
        do_step_3_NS(app)
    if (run_type=="step3graph") or run_all_steps:
        do_step_3_graph(app)
    if (run_type=="step5") or run_all_steps:
        do_step_5(app)
    if (run_type=="step6") or run_all_steps:
        do_step_6(app)
    if (run_type=="step7") or run_all_steps:
        do_step_7(app)
    if (run_type=="step7b") or run_all_steps:
        do_step_7b(app)
    if (run_type=="step8") or run_all_steps:
        do_step_8(app)
    if (run_type=="step9") or run_all_steps:
        do_step_9(app)
    if (run_type=="step10") or run_all_steps:
        do_step_10(app)
