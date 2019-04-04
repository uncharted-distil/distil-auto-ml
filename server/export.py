import os
import json
import yaml
import shutil
import pathlib

from d3m.metadata import pipeline

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


def export(task, rank):
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
    if isinstance(task.pipeline, str):
        pipeline_json = json.loads(task.pipeline)
    else:
        pipeline_json = task.pipeline
    # Set rank
    pipeline_json['pipeline_rank'] = rank
    # Set name
    name = pipeline_json.get('name', False)
    if not name:
        name = task.id        
    pipeline_json['name'] = name
    # Confirm is valid
    #pipeline.PIPELINE_SCHEMA_VALIDATOR.validate(pipeline_json)
    # Write
    pipeline_ranked_dir = pathlib.Path(D3MOUTPUTDIR + '/pipelines_ranked')
    pipeline_ranked_dir.mkdir(parents=True, exist_ok=True)
    pipeline_file = pathlib.Path(pipeline_ranked_dir, '{}.json'.format(task.id))
    with open(pipeline_file, 'w') as f:
        f.write(json.dumps(pipeline_json, sort_keys=True, indent=4))

    # WRITE TO pipelines_scored
    # Write
    pipeline_scored_dir = pathlib.Path(D3MOUTPUTDIR + '/pipelines_scored')
    pipeline_scored_dir.mkdir(parents=True, exist_ok=True)
    scored_file = pathlib.Path(pipeline_scored_dir, '{}.json'.format(task.id))
    with open(scored_file, 'w') as f:
        f.write(json.dumps(pipeline_json, sort_keys=True, indent=4))



def export_run(task, pipeline_run):
    # WRITE TO pipeline_runs
    # Write
    pipeline_runs_dir = pathlib.Path(D3MOUTPUTDIR + '/pipeline_runs')
    pipeline_runs_dir.mkdir(parents=True, exist_ok=True)
    run_file = pathlib.Path(pipeline_runs_dir, '{}.yml'.format(task.id))
    pipeline_run.to_yaml(run_file)

    #pipeline_yaml = yaml.dump(yaml.load(json.dumps(json.loads(task.pipeline_run))), default_flow_style=False)
    #with open(run_file, 'w') as f:
    #    f.write(pipeline_yaml)

