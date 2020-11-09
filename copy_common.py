#!/bin/python3

import glob
import os
import shutil

primitives = [
    "data_transformation.datetime_range_filter.Common",
    "data_transformation.numeric_range_filter.Common",
    "data_transformation.regex_filter.Common",
    "data_transformation.term_filter.Common",
    "data_transformation.datetime_field_compose.Common",
]

json_source = ""
run_source = ""

# find the pipeline / run for common
json_common = glob.glob("./pipelines/common__*.json")
if len(json_common) > 0:
    print(f"Found {json_common[0]}")
    json_source = json_common[0]
else:
    print(f"Pipeline not found - exiting")
    exit(1)

run_common = glob.glob("./pipelines/common__*.gz")
if len(run_common) > 0:
    print(f"Found {run_common[0]}")
    run_source = run_common[0]

json_symlink_source = ""
run_symlink_source = ""

for idx, primitive_path in enumerate(primitives):
    print(f"Processing {primitive_path}")

    json_target = os.path.join("../common-primitives/pipelines", primitive_path)
    json_target_filename = os.path.basename(json_source).replace("common__", "")
    json_target_path = os.path.join(json_target, json_target_filename)
    os.makedirs(json_target, exist_ok=True)
    if idx == 0:
        shutil.copyfile(json_source, json_target_path)
        json_symlink_source = os.path.join("..", primitive_path, json_target_filename)
    elif not os.path.exists(json_target_path):
        os.symlink(json_symlink_source, json_target_path)

    if len(run_source) > 0:
        run_target = os.path.join("../common-primitives/pipeline_runs", primitive_path)
        run_target_filename = os.path.basename(run_source).replace("common__", "")
        run_target_path = os.path.join(run_target, run_target_filename)
        os.makedirs(run_target, exist_ok=True)
        if idx == 0:
            shutil.copyfile(run_source, run_target_path)
            run_symlink_source = os.path.join("..", primitive_path, run_target_filename)
        elif not os.path.exists(run_target_path):
            os.symlink(run_symlink_source, run_target_path)
