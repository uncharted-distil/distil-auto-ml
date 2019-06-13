import os
import sys
import copy
import json
import logging
import importlib

# Make the output a bit quieter...
l = logging.getLogger()
l.addHandler(logging.NullHandler())

# Dir to save files in
META_DIR = 'pipelines'

# Map of default datasets to configure .meta files
# and metric for pipeline config
PIPE_TO_DATASET = {
    'tabular': ('185_baseball', 'f1Macro'),
    'audio': ('31_urbansound', 'accuracy'),
    'clustering': ('1491_one_hundred_plants_margin_clust', 'normalizedMutualInformation'),
    'collaborative_filtering': ('60_jester', 'meanAbsoluteError'),
    'community_detection': ('6_70_com_amazon', 'normalizedMutualInformation'),
    'graph_matching': ('49_facebook', 'accuracy'),
    'image': ('22_handgeometry', 'meanSquaredError'),
    'link_prediction': ('59_umls', 'accuracy'),
    'question_answer': ('32_wikiqa', 'f1'),
    'text': ('30_personae', 'f1'),
    'timeseries_forecasting': ('56_sunspots_monthly', 'rootMeanSquaredError'),
    'timeseries_classification': ('LL1_50words', 'f1Macro'),
    'vertex_nomination': ('LL1_net_nomination_seed', 'accuracy'),
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


if __name__ == '__main__':
    # List all the pipelines
    PIPELINES_DIR = 'processing/pipelines'
    pipelines = [f for f in os.listdir(PIPELINES_DIR) if '.py' in f]

    # For each pipeline, load it and export it
    for pipe in pipelines:
        p = pipe.replace('.py', '')
        print("Handling {}...".format(p))        
        lib = importlib.import_module('processing.pipelines.' + p)
        try:
            # Get the pipeline json and id, write it out
            dataset_to_use, metric = PIPE_TO_DATASET[p]
            pipe_json = lib.create_pipeline(metric=metric).to_json_structure()
            id = pipe_json['id']
            with open(os.path.join(META_DIR, id + '.json'), 'w') as f:
                    f.write(json.dumps(pipe_json, indent=4))
                    f.write('\n')
            # Get meta alongside pipeline
            meta = set_meta(dataset_to_use)
            with open(os.path.join(META_DIR, id + '.meta'), 'w') as f:
                f.write(json.dumps(meta, indent=4))
                f.write('\n')
        except Exception as e:
            print(e)
            sys.exit(0)