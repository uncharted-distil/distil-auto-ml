#!/python yo
import os
import glob
import json
import shutil

# Load the pipelines & meta files
pipeline_files = glob.glob('pipelines/*.json')
pipelines = {}
for p in pipeline_files:
    with open(p) as f:
        id = p.split('/')[-1].replace('py', '')
        pipelines[id] = json.load(f)

meta_files = glob.glob('pipelines/*.meta')

# Load all the primitives
primitive_files = glob.glob('../distil-primitives/annotations/*.json')
primitives = {}
for p in primitive_files:
    # Create a copy of the file in the appropriate place
    # Load the file
    with open(p) as f:
        key = p.split('/')[-1].replace('.json', '')
        annotation = json.load(f)
        id = annotation['id']
        primitives[key] = annotation

    folder_path = os.path.join(
        '../primitives',
        'v' + annotation['primitive_code']['interfaces_version'],
        annotation['source']['name'],
        annotation['python_path'],
        annotation['version']
    )
    primitive_path = os.path.join(folder_path, 'primitive.json')
    print("Moving {} to {}".format(p, primitive_path))
    dir_name, _ = os.path.split(primitive_path)
    os.makedirs(dir_name, exist_ok=True)
    shutil.copy(p, primitive_path)
    for k,v in pipelines.items():
        for s in v['steps']:
            pipe_prim = s['primitive']['python_path']
            if pipe_prim == key:
                old_pipeline_path = [p for p in pipeline_files if k in p][0]
                new_pipeline_path = os.path.join(folder_path, 'pipelines', k)
                old_meta_path = [p for p in meta_files if k.replace('.json', '') in p][0]
                new_meta_path = os.path.join(folder_path, 'pipelines', k.replace('json', 'meta'))
                dir_name, _ = os.path.split(new_pipeline_path)
                os.makedirs(dir_name, exist_ok=True)
                shutil.copy(old_pipeline_path, new_pipeline_path)
                shutil.copy(old_meta_path, new_meta_path)