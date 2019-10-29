import os
import sys
import copy
import json
import logging
import importlib
import hashlib
import traceback

# Make the output a bit quieter...
l = logging.getLogger()
l.addHandler(logging.NullHandler())

# Dir to save files in
META_DIR = 'pipelines'

# Map of default datasets to configure .meta files
# and metric for pipeline config
PIPE_TO_DATASET = {
    'tabular': ('185_baseball', 'f1Macro', {}),
    'audio': ('31_urbansound', 'accuracy', {}),
    'clustering': ('1491_one_hundred_plants_margin_clust', 'normalizedMutualInformation', {
        'num_clusters': 100,
        'cluster_col_name': 'Class'
    }),
    'collaborative_filtering': ('60_jester', 'meanAbsoluteError', {}),
    'community_detection': ('6_70_com_amazon', 'normalizedMutualInformation', {}),
    'graph_matching': ('49_facebook', 'accuracy', {}),
    'image': ('22_handgeometry', 'meanSquaredError', {}),
    'object_detection': ('LL1_penn_fudan_pedestrian', 'objectDetectionAP', {}),
    'link_prediction': ('59_umls', 'accuracy', {}),
    'question_answer': ('32_wikiqa', 'f1', {}),
    'text': ('30_personae', 'f1', {}),
    'timeseries_forecasting': ('56_sunspots_monthly', 'rootMeanSquaredError', {}),
    'timeseries_classification': ('LL1_50words', 'f1Macro', {}),
    'timeseries_kanine': ('LL1_50words', 'f1Macro', {}),
    'timeseries_var': ('56_sunspots', 'rootMeanSquaredError', {}),
    'vertex_nomination': ('LL1_net_nomination_seed', 'accuracy', {}),
    'vertex_classification': ('LL1_VTXC_1369_synthetic', 'f1Macro', {}),
    'semisupervised_tabular': ('SEMI_1040_sylva_prior', 'f1Macro', {}),
    'timeseries_lstm_fcn': ('LL1_50words', 'f1Macro', {}),
    'data_augmentation_tabular': ('DA_ny_taxi_demand', 'meanAbsoluteError', {})
}

# Skeleton of .meta data
META = {
    "problem": "{}_problem",
    "full_inputs": [
        "{}_dataset"
    ],
    "train_inputs": [
        "{}_dataset_TRAIN"
    ],
    "test_inputs": [
        "{}_dataset_TEST"
    ],
    "score_inputs": [
        "{}_dataset_SCORE"
    ]
}

# Gotta update that meta somehow
def set_meta(dataset):
    meta = copy.deepcopy(META)
    for k, v in meta.items():
        if type(v) == str:
            meta[k] = meta[k].format(dataset)
        elif type(v) == list:
            meta[k][0] = meta[k][0].format(dataset)
    return meta


def generate_hash(pipe_json):
    # generate a hash from invariant pipeline data
    pipe_json_mod = copy.deepcopy(pipe_json)
    pipe_json_mod['id'] = ''
    pipe_json_mod['created'] = ''
    pipe_json_mod['digest'] = ''
    return hashlib.md5(json.dumps(pipe_json_mod,sort_keys=True).encode('utf8')).hexdigest()


def generate_file_info():
    print('Generating hashes of existing files....')
    # open the existing pipeline dir and generate hashes for each
    files = [f for f in os.listdir(META_DIR) if '.json' in f]
    hashes = set()
    pipeline_filenames = {}
    for f in files:
        # generate a hash for the file
        with open(os.path.join(META_DIR, f)) as json_data:
            hashes.add(generate_hash(json.load(json_data)))

        # grab the pipeline name and map it to the file
        pipeline_name, pipeline_id = f.split('__')
        pipeline_filenames[pipeline_name] = os.path.join(META_DIR, f.replace('.json', ''))
    return hashes, pipeline_filenames

if __name__ == '__main__':
    # create a hash of the existing invariant pipeline file contents, and a map
    # of pipeline names to pipeline file names
    pipeline_hashes, pipeline_filenames = generate_file_info()

    # List all the pipelines
    PIPELINES_DIR = 'processing/pipelines'
    pipelines = [f for f in os.listdir(PIPELINES_DIR) if '.py' in f]

    # For each pipeline, load it and export it
    for pipe in pipelines:
        p = pipe.replace('.py', '')
        print("Handling {}...".format(p))
        lib = importlib.import_module('processing.pipelines.' + p)
        try:
            dataset_to_use, metric, hyperparams = PIPE_TO_DATASET[p]
            pipe_json = lib.create_pipeline(metric=metric, **hyperparams).to_json_structure()

            hash = generate_hash(pipe_json)
            print(f'Hash for {pipe}: {hash}')
            if hash in pipeline_hashes:
                print(f'Skipping unchanged pipeline for {pipe}')
                continue

            id = pipe_json['id']
            filename = p + '__' + id

            # check if there are existing files asssociated with this pipeline type
            # and delete
            if p in pipeline_filenames:
                os.remove(os.path.join(pipeline_filenames[p] + '.json'))
                os.remove(os.path.join(pipeline_filenames[p] + '.meta'))

            json_filename = os.path.join(META_DIR, filename + '.json')
            meta_filename = os.path.join(META_DIR, filename + '.meta')

            print(f'Writing {filename}')

            with open(json_filename, 'w') as f:
                    f.write(json.dumps(pipe_json, indent=4))
                    f.write('\n')
            # Get meta alongside pipeline
            meta = set_meta(dataset_to_use)
            with open(meta_filename, 'w') as f:
                f.write(json.dumps(meta, indent=4))
                f.write('\n')
        except Exception as e:
            traceback.print_exc()
            sys.exit(0)
