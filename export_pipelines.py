import os
import stat
import sys
import copy
import json
import logging
import importlib
import hashlib
import traceback
import pprint

# Make the output a bit quieter...
l = logging.getLogger()
l.addHandler(logging.NullHandler())


class BCOLORS:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# Dir to save files in
META_DIR = "pipelines"

# Map of default datasets to configure .meta files
# and metric for pipeline config.  This is intended to cover the full set of pipelines.
PIPE_TO_DATASET = {
    "tabular": (
        "185_baseball_MIN_METADATA",
        "f1Macro",
        {
            "profiler": "simple",
            "use_boost": False,
            "grid_search": True,
            "compute_confidences": True,
        },
    ),
    "audio": ("31_urbansound_MIN_METADATA", "accuracy", {}),
    "clustering": (
        "1491_one_hundred_plants_margin_clust_MIN_METADATA",
        "normalizedMutualInformation",
        {"num_clusters": 100, "cluster_col_name": "Class", "profiler": "simon"},
    ),
    "collaborative_filtering": (
        "60_jester_MIN_METADATA",
        "meanAbsoluteError",
        {"min_meta": True},
    ),
    "community_detection": (
        "6_70_com_amazon_MIN_METADATA",
        "normalizedMutualInformation",
        {},
    ),
    "graph_matching": ("49_facebook_MIN_METADATA", "accuracy", {}),
    "image": ("22_handgeometry_MIN_METADATA", "meanSquaredError", {}),
    "object_detection": (
        "LL1_penn_fudan_pedestrian_MIN_METADATA",
        "objectDetectionAP",
        {},
    ),
    "link_prediction": ("59_umls_MIN_METADATA", "accuracy", {}),
    "question_answer": ("32_wikiqa_MIN_METADATA", "f1", {}),
    "text": ("30_personae_MIN_METADATA", "f1", {}),
    "text_sent2vec": ("LL1_TXT_CLS_3746_newsgroup_MIN_METADATA", "accuracy", {}),
    "timeseries_forecasting": (
        "56_sunspots_monthly_MIN_METADATA",
        "rootMeanSquaredError",
        {},
    ),
    "timeseries_kanine": ("LL1_50words_MIN_METADATA", "f1Macro", {}),
    "timeseries_var": (
        "LL1_736_population_spawn_simpler_MIN_METADATA",
        "meanAbsoluteError",
        {},
    ),
    "timeseries_nbeats": (
        "LL1_736_population_spawn_simpler_MIN_METADATA",
        "meanAbsoluteError",
        {},
    ),
    "vertex_nomination": ("LL1_net_nomination_seed_MIN_METADATA", "accuracy", {}),
    "vertex_nomination": (
        "LL1_EDGELIST_net_nomination_seed_MIN_METADATA",
        "accuracy",
        {"is_edgelist": True},
    ),
    "vertex_classification": ("LL1_VTXC_1369_synthetic_MIN_METADATA", "f1Macro", {}),
    "semisupervised_tabular": ("SEMI_1040_sylva_prior_MIN_METADATA", "f1Macro", {}),
    "timeseries_lstm_fcn": ("LL1_50words_MIN_METADATA", "f1Macro", {}),
    "mi_ranking": ("185_baseball_MIN_METADATA", "f1Macro", {}),
    "remote_sensing": (
        "LL1_bigearth_landuse_detection",
        "f1Macro",
        {"svc": True, "confidences": False},
    ),
    "remote_sensing_mlp": ("LL1_bigearth_landuse_detection", "f1Macro", {}),
    "common": ("LL0_acled_reduced_MIN_METADATA", "f1Macro", {}),
    "remote_sensing_filtered": ("LL1_bigearth_landuse_detection", "f1Macro", {}),
}

# Subset of pipelines that are aimed at coverage of only the primitives that we intend to
# submit to the d3m repo.
SUBMISSION_SUBSET = set(
    [
        "tabular",
        "audio",
        "clustering",
        "collaborative_filtering",
        "community_detection",
        "graph_matching",
        "image",
        "link_prediction",
        "question_answer",
        "text",
        "vertex_nomination",
        "remote_sensing",
        "timeseries_var",  # covers timeseries binning
        "timeseries_kanine",  # covers timeseries formatter
        "common",
        "mi_ranking",
        "remote_sensing_filtered",  # covers vector bounds filter
    ]
)


def generate_hash(pipe_json):
    # generate a hash from invariant pipeline data
    pipe_json_mod = copy.deepcopy(pipe_json)
    pipe_json_mod["id"] = ""
    pipe_json_mod["created"] = ""
    pipe_json_mod["digest"] = ""
    return hashlib.md5(
        json.dumps(pipe_json_mod, sort_keys=True).encode("utf8")
    ).hexdigest()


def generate_file_info():
    print("Generating hashes of existing files....")
    # open the existing pipeline dir and generate hashes for each
    files = [f for f in os.listdir(META_DIR) if ".json" in f and not "_run" in f]
    hashes = set()
    pipeline_filenames = {}
    for f in files:
        # generate a hash for the file
        with open(os.path.join(META_DIR, f)) as json_data:
            hashes.add(generate_hash(json.load(json_data)))

        # grab the pipeline name and map it to the file
        pipeline_name, pipeline_id = f.split("__")
        pipeline_filenames[pipeline_name] = os.path.join(
            META_DIR, f.replace(".json", "")
        )
    return hashes, pipeline_filenames


def strip_digests(pipeline_json):
    for step in pipeline_json["steps"]:
        del step["primitive"]["digest"]


if __name__ == "__main__":
    submission_only = False
    if len(sys.argv) > 1:
        submission_only = sys.argv[1] == "--submission"

    print(f"Submission only: {submission_only}")

    # create a hash of the existing invariant pipeline file contents, and a map
    # of pipeline names to pipeline file names
    pipeline_hashes, pipeline_filenames = generate_file_info()

    # List all the pipelines
    PIPELINES_DIR = "processing/pipelines"
    VALIDATION_PIPELINES_DIR = "processing/validation_pipelines"
    pipelines = [
        f
        for f in os.listdir(PIPELINES_DIR) + os.listdir(VALIDATION_PIPELINES_DIR)
        if ".py" in f
    ]

    # For each pipeline, load it and export it
    for pipe in pipelines:
        p = pipe.replace(".py", "")

        if submission_only and p not in SUBMISSION_SUBSET:
            continue

        # if we're generating pipelines for submission only, use the subset that covers
        # our primitives only
        print("Handling {}...".format(p))

        try:
            try:
                lib = importlib.import_module("processing.pipelines." + p)
            except ModuleNotFoundError:
                lib = importlib.import_module("processing.validation_pipelines." + p)
            dataset_to_use, metric, hyperparams = PIPE_TO_DATASET[p]
            pipe_obj = lib.create_pipeline(metric=metric, **hyperparams)
            pipe_json = pipe_obj[0].to_json_structure()
            strip_digests(pipe_json)

            hash = generate_hash(pipe_json)
            print(f"Hash for {pipe}: {hash}")
            if hash in pipeline_hashes:
                print(
                    f"{BCOLORS.WARNING} Skipping unchanged pipeline for {pipe}{BCOLORS.ENDC}"
                )
                continue

            id = pipe_json["id"]
            filename = p + "__" + id

            # check if there are existing files asssociated with this pipeline type
            # and delete
            if p in pipeline_filenames:
                os.remove(os.path.join(pipeline_filenames[p] + ".json"))
                os.remove(os.path.join(pipeline_filenames[p] + ".sh"))

            json_filename = os.path.join(META_DIR, filename + ".json")
            run_filename = os.path.join(META_DIR, filename + ".sh")
            output_filename = os.path.join(META_DIR, filename + "_run.yaml")

            print(f"{BCOLORS.OKGREEN}Writing {filename}{BCOLORS.ENDC}")

            with open(json_filename, "w") as f:
                f.write(json.dumps(pipe_json, indent=4))
                f.write("\n")

            # account for the d3m datasets being organized such that clustering is off in its own directory
            dataset_prob_dir = dataset_to_use
            if p == "clustering":
                dataset_to_use = f"../seed_datasets_unsupervised/{dataset_to_use}"

            # Generate a convenience run file
            runtime_args = f"-v $D3MSTATICDIR -d $D3MINPUTDIR"
            problem_arg = f"-r $D3MINPUTDIR/{dataset_to_use}/{dataset_prob_dir}_problem/problemDoc.json"
            train_arg = f"\t-i $D3MINPUTDIR/{dataset_to_use}/TRAIN/dataset_TRAIN/datasetDoc.json"
            test_arg = (
                f"-t $D3MINPUTDIR/{dataset_to_use}/TEST/dataset_TEST/datasetDoc.json"
            )
            if os.path.isfile(
                os.path.expandvars(
                    "$D3MINPUTDIR/"
                    + dataset_to_use
                    + "/SCORE/dataset_TEST/datasetDoc.json"
                )
            ):
                score_arg = f"-a $D3MINPUTDIR/{dataset_to_use}/SCORE/dataset_TEST/datasetDoc.json"
            else:
                score_arg = f"-a $D3MINPUTDIR/{dataset_to_use}/SCORE/dataset_SCORE/datasetDoc.json"
            pipeline_arg = f"-p {json_filename}"
            output_arg = f"-O {output_filename}"
            fit_score_args = f"{problem_arg} {train_arg} {test_arg} {score_arg} {pipeline_arg} {output_arg}"

            with open(run_filename, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(
                    f"python3 -m d3m runtime {runtime_args} fit-score {fit_score_args} && \n"
                )
                f.write(f"gzip -f {output_filename}")
                f.write("\n")
            st = os.stat(run_filename)
            os.chmod(run_filename, st.st_mode | stat.S_IEXEC)

        except Exception as e:
            print(e)
            print(f"{BCOLORS.FAIL}Skipping errored pipeline {p}.{BCOLORS.ENDC}")
