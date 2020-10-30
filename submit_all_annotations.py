#!/python yo
import os
import glob
import json
import shutil
import sys


def clean_pipelines(folder_path):
    sub_folders = ("pipelines", "pipeline_runs")
    for sub_folder in sub_folders:
        pathd = os.path.join(folder_path, sub_folder)
        if os.path.isdir(pathd):
            print(f"Deleting {pathd}")
            shutil.rmtree(pathd)


annotations_only = False
needs_pipelines = False
if len(sys.argv) > 1:
    annotations_only = sys.argv[1] == "--annotations"
    needs_pipelines = sys.argv[1] == "--needs-pipelines"

# Generate the list of run archive files, and store the set of their ids.
run_files = glob.glob("pipelines/*_run.yaml.gz")
run_ids = set()
for p in run_files:
    id = p.split("/")[-1].replace("_run.yaml.gz", "").split("__")[1]
    run_ids.add(id)

# Load the pipeline files and store the json for lookup by pipeline id.
pipeline_files = glob.glob("pipelines/*.json")
pipelines = {}
for p in pipeline_files:
    with open(p) as f:
        id = p.split("/")[-1].replace(".json", "").split("__")[1]
        if id in run_ids:
            pipelines[id] = json.load(f)

# Loop over each primitive that we have a generated annotation for
primitive_files = glob.glob("../distil-primitives/annotations/*.json")
primitives = {}
for p in primitive_files:
    # Create a copy of the file in the appropriate place

    # Load the primitive annotation file
    with open(p) as f:
        key = p.split("/")[-1].replace(".json", "")
        annotation = json.load(f)
        primitives[key] = annotation

    # Figure out the target folder hierarchy based on the metadata - this will be in the
    # sibling `primitives` project
    folder_path = os.path.join(
        "../primitives",
        "primitives",
        annotation["source"]["name"],
        annotation["python_path"],
        annotation["version"],
    )

    # clean existing pipelines out of the `primitives` pipeline folders to avoid accumulating
    clean_pipelines(folder_path)

    # loop through primitives in each pipeline to find one that matches our current annotation
    if not annotations_only:
        pipeline_match = False
        for pipeline_id, pipeline_json in pipelines.items():
            for step in pipeline_json["steps"]:
                pipe_prim = step["primitive"]["python_path"]
                if pipe_prim == key:
                    # this pipeline has a primitive step that uses the current annotation
                    pipeline_match = True

                    # Incoming filenames are <pipeline type>__<pipeline id>.json
                    # Format for deploy is <pipeline id>.json

                    old_pipeline_path = [p for p in pipeline_files if pipeline_id in p][
                        0
                    ]
                    new_pipeline_path = os.path.join(
                        folder_path, "pipelines", pipeline_id + ".json"
                    )

                    old_run_path = [p for p in run_files if pipeline_id in p][0]
                    new_run_path = os.path.join(
                        folder_path, "pipeline_runs", pipeline_id + "_run.yaml.gz"
                    )

                    if needs_pipelines and (
                        not os.path.exists(old_run_path)
                        or not os.path.exists(old_pipeline_path)
                    ):
                        continue

                    dir_name, _ = os.path.split(new_pipeline_path)
                    os.makedirs(dir_name, exist_ok=True)

                    dir_name, _ = os.path.split(new_run_path)
                    os.makedirs(dir_name, exist_ok=True)

                    # only copy one pipeline file for any given primitive
                    pipeline_root_path = os.path.join(folder_path, "pipelines")
                    if os.path.isdir(pipeline_root_path) and not os.listdir(
                        pipeline_root_path
                    ):
                        shutil.copy(old_pipeline_path, new_pipeline_path)

                    run_root_path = os.path.join(folder_path, "pipeline_runs")
                    if os.path.isdir(run_root_path) and not os.listdir(run_root_path):
                        shutil.copy(old_run_path, new_run_path)

    # copy the primitive annotation file into the target folder in the `primitives` project
    primitive_path = os.path.join(folder_path, "primitive.json")
    print("Moving {} to {}".format(p, primitive_path))
    dir_name, _ = os.path.split(primitive_path)
    os.makedirs(dir_name, exist_ok=True)
    shutil.copy(p, primitive_path)
