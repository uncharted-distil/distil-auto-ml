from processing.pipelines import tabular
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# CDB: EVAL ONLY WORKAROUND
    # Our one hot encoder fails when there are categorical values present but one are below
    # the encoding threshold, and the SKLearnWrap one hot encoder fails in any case that you
    # pass it a dataframe with no categorical columns for it to encode.  The MS Geolife dataset
    # has categorical columns that are all binary encoded so the pipeline fails to work in that case.
    # Since primitives are frozen for the eval, the only thing we can do is threshold check the categoricals
    # remove the categorical primitive if none pass.
    
def create_pipeline(metric: str) -> Pipeline:
    return tabular.create_pipeline(metric, include_one_hot=False)