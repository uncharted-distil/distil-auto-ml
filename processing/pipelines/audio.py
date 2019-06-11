import sys
from typing import List, Dict, Any, Tuple, Set
import logging
import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.base import ArgumentType



from common_primitives.construct_predictions import ConstructPredictionsPrimitive

from distil.primitives.ensemble_forest import EnsembleForestPrimitive
from distil.primitives.audio_transfer import AudioTransferPrimitive

from distil.primitives.audio_loader import AudioDatasetLoaderPrimitive


PipelineContext = utils.Enum(value='PipelineContext', names=['TESTING'], start=1)

# CDB: Totally unoptimized.  Pipeline creation code could be simplified but has been left
# in a naively implemented state for readability for now.
#
# Overall implementation relies on passing the entire dataset through the pipeline, with the primitives
# identifying columns to operate on based on type.  Alternative implementation (that better lines up with
# D3M approach, but generates more complex pipelines) would be to extract sub-sets by semantic type using
# a common primitive, apply the type-specific primitive to the sub-set, and then merge the changes
# (replace or join) back into the original data.
def create_pipeline(metric: str,
                    cat_mode: str = 'one_hot',
                    max_one_hot: int = 16,
                    scale: bool = False) -> Pipeline:
    previous_step = 0
    input_val = 'steps.{}.produce'

    # create the basic pipeline
    audio_pipeline = Pipeline(context=PipelineContext.TESTING)
    audio_pipeline.add_input(name='inputs')


    # step 0
    step = PrimitiveStep(primitive_description=AudioDatasetLoaderPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    step.add_output('produce_target')
    step.add_output('produce_collection')
    audio_pipeline.add_step(step)


    # step 1 - featurize
    step = PrimitiveStep(primitive_description=AudioTransferPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce_collection')
    step.add_output('produce')
    audio_pipeline.add_step(step)


    # step 2 -- Generates a random forest ensemble model.
    step = PrimitiveStep(primitive_description=EnsembleForestPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce_target')
    step.add_output('produce')
    step.add_hyperparameter('metric', ArgumentType.VALUE, metric)
    step.add_hyperparameter('fast', ArgumentType.VALUE, False) # turn off, test dataset too small
    audio_pipeline.add_step(step)


    # step 3 - convert predictions to expected format
    step = PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
    step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce_target')
    step.add_output('produce')
    step.add_hyperparameter('use_columns', ArgumentType.VALUE, [0, 1])
    audio_pipeline.add_step(step)


    # Adding output step to the pipeline
    audio_pipeline.add_output(name='output', data_reference='steps.3.produce')

    return audio_pipeline
