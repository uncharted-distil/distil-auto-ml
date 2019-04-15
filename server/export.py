import os
import json
import yaml
import shutil
import pathlib

from d3m.metadata import pipeline

import main_utils as utils

import config

# eval mode, oneof {search, test, ta2ta3}
D3MRUN = os.getenv("D3MRUN", "invalid")

# directory setup
D3MINPUTDIR = os.getenv("D3MINPUTDIR", "/input")
D3MOUTPUTDIR = os.getenv("D3MOUTPUTDIR", "/output")

# eval constraints
D3MCPU = os.getenv("D3MCPU", None)
D3MRAM = os.getenv("D3MRAM", None)
D3MTIMEOUT = os.getenv("D3MTIMEOUT", None)

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


def export_predictions(solution_task):
    # copy predictions from original scoring output path to the expected D3M location
    solution_results_dir = pathlib.Path(D3MOUTPUTDIR, 'predictions', str(solution_task.id))
    solution_results_dir.mkdir(parents=True, exist_ok=True)
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

        pipelines_ranked - a directory with ranked pipelines to be evaluated, named <pipeline id>.json; these files should have additional field pipeline_rank
        pipelines_scored - a directory with successfully scored pipelines during the search, named <pipeline id>.json
        pipeline_runs - a directory with pipeline run records in YAML format, multiple can be stored in the same file, named <pipeline run id>.yml
    """
    # WRITE TO pipelines_ranked
    # ensure it is proper JSON by loading it first
    if isinstance(solution_task.pipeline, str):
        pipeline_json = json.loads(solution_task.pipeline)
    else:
        pipeline_json = fit_solution_task.pipeline
    # Set rank
    pipeline_json['pipeline_rank'] = rank
    # Set name
    name = pipeline_json.get('name', False)
    if not name:
        name = solution_task.id
    pipeline_json['name'] = name
    # Confirm is valid
    #pipeline.PIPELINE_SCHEMA_VALIDATOR.validate(pipeline_json)
    # Write
    pipeline_ranked_dir = pathlib.Path(D3MOUTPUTDIR + '/pipelines_ranked')
    pipeline_ranked_dir.mkdir(parents=True, exist_ok=True)
    pipeline_file = pathlib.Path(pipeline_ranked_dir, '{}.json'.format(solution_task.id))
    with open(pipeline_file, 'w') as f:
        f.write(json.dumps(pipeline_json, sort_keys=True, indent=4))

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

