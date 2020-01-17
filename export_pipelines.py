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

# Dir to save files in
META_DIR = 'pipelines'

# Map of default datasets to configure .meta files
# and metric for pipeline config
PIPE_TO_DATASET = {
    'tabular': ('LL0_acled_reduced_MIN_METADATA', 'f1Macro', {}),
    'audio': ('31_urbansound', 'accuracy', {}),
    'clustering': ('1491_one_hundred_plants_margin_clust', 'normalizedMutualInformation', {
        'num_clusters': 100,
        'cluster_col_name': 'Class'
    }),
    'collaborative_filtering': ('60_jester_MIN_METADATA', 'meanAbsoluteError', {}),
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
    'timeseries_var': ('LL1_736_population_spawn_simpler', 'meanAbsoluteError', {}),
    'timeseries_deepar': ('LL1_736_population_spawn', 'meanAbsoluteError', {}),
    'vertex_nomination': ('LL1_net_nomination_seed', 'accuracy', {}),
    'vertex_classification': ('LL1_VTXC_1369_synthetic', 'f1Macro', {}),
    'semisupervised_tabular': ('SEMI_1040_sylva_prior', 'f1Macro', {}),
    'timeseries_lstm_fcn': ('LL1_50words', 'f1Macro', {}),
    'data_augmentation_tabular': ('DA_ny_taxi_demand', 'meanAbsoluteError', {})
}

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
    files = [f for f in os.listdir(META_DIR) if '.json' in f and not '_run' in f]
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

def strip_digests(pipeline_json):
    for step in pipeline_json['steps']:
        del step['primitive']['digest']

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
        try:
            lib = importlib.import_module('processing.pipelines.' + p)
            dataset_to_use, metric, hyperparams = PIPE_TO_DATASET[p]
            pipe_obj = lib.create_pipeline(metric=metric, **hyperparams)
            pipe_json = pipe_obj.to_json_structure()
            strip_digests(pipe_json)

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
                os.remove(os.path.join(pipeline_filenames[p] + '.sh'))

            json_filename = os.path.join(META_DIR, filename + '.json')
            run_filename = os.path.join(META_DIR, filename + '.sh')
            output_filename = os.path.join(META_DIR, filename + '_run.yaml')

            print(f'Writing {filename}')

            with open(json_filename, 'w') as f:
                    f.write(json.dumps(pipe_json, indent=4))
                    f.write('\n')

            # Generate a convenience run file
            runtime_args = f'-v $D3MSTATICDIR -d $D3MINPUTDIR'
            problem_arg = f'-r $D3MINPUTDIR/{dataset_to_use}/{dataset_to_use}_problem/problemDoc.json'
            train_arg = f'\t-i $D3MINPUTDIR/{dataset_to_use}/TRAIN/dataset_TRAIN/datasetDoc.json'
            test_arg = f'-t $D3MINPUTDIR/{dataset_to_use}/TEST/dataset_TEST/datasetDoc.json'
            score_arg = f'-a $D3MINPUTDIR/{dataset_to_use}/SCORE/dataset_SCORE/datasetDoc.json'
            pipeline_arg = f'-p {json_filename}'
            output_arg = f'-O {output_filename}'
            fit_score_args = f'{problem_arg} {train_arg} {test_arg} {score_arg} {pipeline_arg} {output_arg}'

            with open(run_filename, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write(f'python3 -m d3m runtime {runtime_args} fit-score {fit_score_args} && \n')
                f.write(f'gzip -f {output_filename}')
                f.write('\n')
            st = os.stat(run_filename)
            os.chmod(run_filename, st.st_mode | stat.S_IEXEC)

        except Exception as e:
            print(e)
            print(f'Skipping errored pipeline {p}.')
