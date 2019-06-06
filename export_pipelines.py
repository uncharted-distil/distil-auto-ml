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
PIPE_TO_DATASET = {
    'tabular': '185_baseball',
    'audio': '31_urbansound',
    'clustering': '1491_one_hundred_plants_margin_clust',
    'collaborative_filtering': '60_jester',
    'community_detection': '6_70_com_amazon',
    'graph_matching': '49_facebook',
    'image': '22_handgeometry',
    'link_prediction': '59_umls',
    'question_answer': '32_wikiqa',
    'text': '30_personae',
    'timeseries': '56_sunspots_monthly',
    'vertex_nomination': 'LL1_net_nomination_seed',
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
    meta = copy.copy(META)
    for k, v in meta.items():
        if type(v) == str:
            meta[k] = meta[k].format(dataset)
        if type(v) == list:
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
            pipe_json = lib.create_pipeline(metric='f1Macro').to_json_structure()
            id = pipe_json['id']
            with open(os.path.join(META_DIR, id + '.json'), 'w') as f:
                    f.write(json.dumps(pipe_json, indent=4))
                    f.write('\n')
            # Get meta alongside pipeline
            dataset_to_use = PIPE_TO_DATASET[p]
            meta = set_meta(dataset_to_use)
            with open(os.path.join(META_DIR, id + '.meta'), 'w') as f:
                f.write(json.dumps(meta, indent=4))
                f.write('\n')
        except Exception as e:
            print(e)
            sys.exit(0)