import os
import json
import shutil
import pathlib

from d3m.metadata import pipeline

import config


def get_executable_solution_id(test_opt_input):
    """Return the solution id of the solution/pipeline to test on.

    test_opt_input is from the D3MTESTOPT env var and is guaranteed to be the
    full path and filename for the executable we are to test on.

    The naming convention we use when writing our output is `<solution_id>.<extension>`
    so we can extract just the filename to get the solution_id.
    """
    soln_id = pathlib.Path(test_opt_input).resolve().stem
    return soln_id


def override_config():
    supporting_files_dir = pathlib.Path(D3MOUTPUTDIR, "supporting_files")
    supporting_files_dir.mkdir(parents=True, exist_ok=True)

    # override aretha output directory to be supporting_files directory
    # which is always preserved between runs, meaning we'll
    # always have access to files written to this directory
    config.OUTPUT_DIR = str(supporting_files_dir.resolve())
    # override log filename so it is saved properly and preserved between runs
    config.LOG_FILENAME = str(pathlib.Path(supporting_files_dir, "eval_log.log").resolve())
    # override db filename so it is saved properly and preserved between runs
    config.DB_LOCATION = str(pathlib.Path(supporting_files_dir, "eval_db.db").resolve())
    # put system in debug mode always
    config.DEBUG = True


# eval mode, oneof {search, test, ta2ta3}
D3MRUN = os.getenv("D3MRUN", "invalid")

# directory setup
D3MINPUTDIR = os.getenv("D3MINPUTDIR", "/input")
D3MOUTPUTDIR = os.getenv("D3MOUTPUTDIR", "/output")

# configs
if D3MRUN == 'search' or D3MRUN == 'ta2ta3':
    # get the path to the config
    CONFIG_PATH = pathlib.Path(D3MINPUTDIR, "search_config.json")

if D3MRUN == 'test':
    # get the path to the config
    CONFIG_PATH = pathlib.Path(D3MINPUTDIR, "test_config.json")

    # get the id of the solution to be tested
    # TODO: figure out how data machines is getting and passing this
    D3MTESTOPT = os.getenv("D3MTESTOPT")
    TEST_FIT_SOLUTION_ID = get_executable_solution_id(D3MTESTOPT)

# eval constraints
D3MCPU = os.getenv("D3MCPU", None)
D3MRAM = os.getenv("D3MRAM", None)
D3MTIMEOUT = os.getenv("D3MTIMEOUT", None)




def export_dag(dag_json, fitted_soln_id, rank):
    # ensure it is proper JSON by loading it first
    if isinstance(dag_json, str):
        dag_json = json.loads(dag_json)

    # Convert to program-style pipe
    dag_json = convert_pipeline(dag_json, fitted_soln_id)

    # Set rank
    dag_json['pipeline_rank'] = rank

    # Set name
    name = dag_json.get('name', False)
    if not name:
        name = fitted_soln_id        
    dag_json['name'] = name

    # Confirm is valid
    pipeline.PIPELINE_SCHEMA_VALIDATOR.validate(dag_json)

    parser = translate.EvalParser(setup.CONFIG_PATH, setup.D3MRUN)
    pipeline_output_dir = pathlib.Path(parser.get_pipeline_logs_dir())
    # TRUST NO ONE
    pipeline_output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_file = pathlib.Path(pipeline_output_dir, '{}.json'.format(fitted_soln_id))
    with open(pipeline_file, 'w') as f:
        f.write(json.dumps(dag_json, sort_keys=True, indent=4))

def export_executable(task_id, fitted_soln_id):
    parser = translate.EvalParser(setup.CONFIG_PATH, setup.D3MRUN)
    executables_dir = pathlib.Path(parser.get_executables_dir())
    # TRUST NO ONE
    executables_dir.mkdir(parents=True, exist_ok=True)

    # task_id is how we reference fitted solutions internally, but use fitted_solution_id for the fn
    # copy over the dill file and rename it
    task_fn = utils.make_job_fn(task_id)
    shutil.copy(task_fn, executables_dir)
    old_fn = pathlib.Path(executables_dir, '{}.dill'.format(task_id))
    executable_file = pathlib.Path(executables_dir, '{}.dill'.format(fitted_soln_id))
    old_fn.rename(executable_file)

def export_predictions(original_preds_path, expected_preds_fn, fitted_soln_id):
    parser = translate.EvalParser(setup.CONFIG_PATH, setup.D3MRUN)
    results_dir = parser.get_predictions_dir()
    solution_results_dir = pathlib.Path(results_dir, str(fitted_soln_id))
    # TRUST NO ONE
    solution_results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(original_preds_path, solution_results_dir)
    task_id = original_preds_path.resolve().stem
    old_fn = pathlib.Path(solution_results_dir, '{}.csv'.format(task_id))
    expected_name = pathlib.Path(solution_results_dir, expected_preds_fn)
    old_fn.rename(expected_name)