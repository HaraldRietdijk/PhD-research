from steps.step_1.step_1_Isala import do_step_1_isala

import sys

def do_run_Isala(app):
    run_type = sys.argv[2]
    run_all_steps = (run_type=="all")
    if (run_type=="step1") or run_all_steps:
        do_step_1_isala(app)
    # if (run_type=="step2") or run_all_steps:
    #     do_step_2_NS(app)
    # if (run_type=="step3") or run_all_steps:
    #     do_step_3_NS(app)
    # if (run_type=="step3graph") or run_all_steps:
    #     do_step_3_graph(app)
    # if (run_type=="step5") or run_all_steps:
    #     do_step_5(app)
    # if (run_type=="step6") or run_all_steps:
    #     do_step_6(app)
    # if (run_type=="step7") or run_all_steps:
    #     do_step_7(app)
    # if (run_type=="step7b") or run_all_steps:
    #     do_step_7b(app)
    # if (run_type=="step8") or run_all_steps:
    #     do_step_8(app)
    # if (run_type=="step9") or run_all_steps:
    #     do_step_9(app)
    # if (run_type=="step10") or run_all_steps:
    #     do_step_10(app)
