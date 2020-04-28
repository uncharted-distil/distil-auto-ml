import os
import json
import pickle
import yaml
import shutil
import pathlib

from d3m.metadata import pipeline

import utils

import config

# eval mode, oneof {search, test, ta2ta3}
D3MRUN = os.getenv("D3MRUN", "invalid")

# directory setup
local_env = os.getenv("D3MLOCAL") # set this to true if not running in docker.
if local_env == 'True':
    D3MINPUTDIR = os.getenv("D3MINPUTDIR", "./input")
    D3MOUTPUTDIR = os.getenv("D3MOUTPUTDIR", "./output")
else:
    D3MINPUTDIR = os.getenv("D3MINPUTDIR", "/input")
    D3MOUTPUTDIR = os.getenv("D3MOUTPUTDIR", "/output")

# eval constraints
D3MCPU = os.getenv("D3MCPU", None)
D3MRAM = os.getenv("D3MRAM", None)
D3MTIMEOUT = os.getenv("D3MTIMEOUT", None)

def override_config():
    supporting_files_dir = pathlib.Path(D3MOUTPUTDIR, "supporting_files")
    supporting_files_dir.mkdir(parents=True, exist_ok=True)

    # override log filename so it is saved properly and preserved between runs
    log_dir = pathlib.Path(D3MOUTPUTDIR, 'logs').resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    config.LOG_FILENAME = str(pathlib.Path(log_dir, 'distil_auto_ml.log').resolve())

    # override aretha output directory to be supporting_files directory
    # which is always preserved between runs, meaning we'll
    # always have access to files written to this directory
    config.OUTPUT_DIR = str(supporting_files_dir.resolve())
    # override db filename so it is saved properly and preserved between runs
    config.DB_LOCATION = str(pathlib.Path(supporting_files_dir, "eval_db.db").resolve())

def export_predictions(solution_task):
    # copy predictions from original scoring output path to the expected D3M location
    solution_results_dir = pathlib.Path(D3MOUTPUTDIR, 'predictions', str(solution_task.id))
    solution_results_dir.mkdir(parents=True, exist_ok=True)
    # TODO solution_task.output_keys is none, but make_preds_filename requires it.
    original_preds_path = utils.make_preds_filename(solution_task.id)
    # check if predictions were actually generated and export them if so
    if os.path.exists(original_preds_path):
        shutil.copy(original_preds_path, solution_results_dir)
        old_fn = pathlib.Path(solution_results_dir, '{}.csv'.format(solution_task.id))
        expected_name = pathlib.Path(solution_results_dir, 'predictions.csv')
        old_fn.rename(expected_name)


def export(solution_task, rank):
    """
    https://datadrivendiscovery.org/wiki/pages/viewpage.action?spaceKey=work&title=Evaluation+Workflow
    Output directory structure

    D3MOUTPUTDIR points to an output directory. Some sub-directories of this directory have a predefined role and structure. All other locations can be used arbitrary for TA2-TA3 communication or for communication with the data mart. (Multiple systems are sharing this read-write directory.) Defined directories:

        pipelines_ranked - a directory with ranked pipelines to be evaluated, named <pipeline id>.json and <pipeline id>.rank.  The rank
                           the rank file is a text file that contains a single value, which is the rank for the pipeline.
        pipelines_scored - a directory with successfully scored pipelines during the search, named <pipeline id>.json
        pipeline_runs - a directory with pipeline run records in YAML format, multiple can be stored in the same file, named <pipeline run id>.yml
    """
    # WRITE TO pipelines_ranked
    pipeline_json = create_json(solution_task)
    # Confirm is valid
    #pipeline.PIPELINE_SCHEMA_VALIDATOR.validate(pipeline_json)

    # Write out pipelines_ranked
    # first the pipeline
    pipeline_ranked_dir = pathlib.Path(D3MOUTPUTDIR + '/pipelines_ranked')
    pipeline_ranked_dir.mkdir(parents=True, exist_ok=True)
    pipeline_file = pathlib.Path(pipeline_ranked_dir, '{}.json'.format(solution_task.id))
    with open(pipeline_file, 'w') as f:
        f.write(json.dumps(pipeline_json, sort_keys=True, indent=4))
    # next the associated rank
    pipeline_file = pathlib.Path(pipeline_ranked_dir, '{}.rank'.format(solution_task.id))
    with open(pipeline_file, 'w') as f:
        f.write(str(rank))

    # WRITE TO pipelines_scored
    # Write
    pipeline_scored_dir = pathlib.Path(D3MOUTPUTDIR + '/pipelines_scored')
    pipeline_scored_dir.mkdir(parents=True, exist_ok=True)
    scored_file = pathlib.Path(pipeline_scored_dir, '{}.json'.format(solution_task.id))
    with open(scored_file, 'w') as f:
        f.write(json.dumps(pipeline_json, sort_keys=True, indent=4))


def export_run(solution_task):
    # WRITE TO pipeline_runs
    # Write
    pipeline_runs_dir = pathlib.Path(D3MOUTPUTDIR + '/pipeline_runs')
    pipeline_runs_dir.mkdir(parents=True, exist_ok=True)
    run_file = pathlib.Path(pipeline_runs_dir, '{}.yml'.format(solution_task.id))
    with open(run_file, 'w') as f:
        f.write(solution_task.pipeline_run)

def save_pipeline(solution_task):
    pipeline_json = create_json(solution_task)
    pipeline_dir = pathlib.Path(D3MOUTPUTDIR, 'pipelines')
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    filename = pathlib.Path(pipeline_dir, f'{solution_task.id}.json').resolve()
    with open(filename, 'w') as f:
        f.write(json.dumps(pipeline_json, sort_keys=True, indent=4))

    return filename.as_uri()

def save_fitted_pipeline(fitted_solution_id, runtime):
    pipeline_dir = pathlib.Path(D3MOUTPUTDIR, 'fitted_pipelines')
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    filename = pathlib.Path(pipeline_dir, f'{fitted_solution_id}.d3m').resolve()
    with open(filename, 'wb') as pickle_file:
        pickle.dump(runtime, pickle_file)

    return filename.as_uri()

def create_json(solution_task):
    # ensure it is proper JSON by loading it first
    if isinstance(solution_task.pipelines, str):
        pipeline_json = json.loads(solution_task.pipelines)
    else:
        pipeline_json = solution_task.pipelines
    # Set name
    name = pipeline_json.get('name', False)
    if not name:
        name = solution_task.id
    pipeline_json['name'] = name

    return pipeline_json
