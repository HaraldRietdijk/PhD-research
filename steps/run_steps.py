from steps.read_raw_data.read_raw_data import do_read_raw_data
from steps.step_1.step_1 import do_step_1
from steps.step_2.step_2 import do_step_2
from steps.step_3.step_3 import do_step_3
from steps.step_3.step_3_graph import do_step_3_graph
from steps.step_4.step_4 import do_step_4
from steps.step_5.step_5 import do_step_5
from steps.step_6.step_6 import do_step_6
from steps.step_7.step_7 import do_step_7
from steps.step_7.step_7b import do_step_7b
from simulator.simulator_run import do_simulator_run
from examples_and_trials.tests import do_test
import sys

def do_run(app):
    run_type = sys.argv[2]
    run_all = (run_type=="all")
    if (run_type=="rawdata"):
        do_read_raw_data(app)
    if (run_type=="step1") or run_all:
        do_step_1(app)
    if (run_type=="step2") or run_all:
        do_step_2(app)
    if (run_type=="step3") or run_all:
        do_step_3(app)
    if (run_type=="step3graph") or run_all:
        do_step_3_graph(app)
    if (run_type=="step4") or run_all:
        do_step_4(app)
    if (run_type=="step5") or run_all:
        do_step_5(app)
    if (run_type=="step6") or run_all:
        do_step_6(app)
    if (run_type=="step7") or run_all:
        do_step_7(app)
    if (run_type=="step7b") or run_all:
        do_step_7b(app)
    if (run_type=="sim"):
        do_simulator_run(app)
    if (run_type=="test"):
        do_test(app)
