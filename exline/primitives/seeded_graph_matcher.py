import os
import logging
from typing import Set, List, Dict, Any, Optional

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from d3m.primitive_interfaces.base import CallResult

import pandas as pd
import numpy as np
import networkx as nx
from scipy import sparse

from sgm.backends.classic import ScipyJVClassicSGM

__all__ = ('SeededGraphMatcher',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    metric = hyperparams.Hyperparameter[str](
        default='',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class Params(params.Params):
    pass

class ExlineSeededGraphMatchingPrimitive(PrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    A primitive that matches seeded graphs.
    """
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '8baea8e6-9d3a-46d7-acf1-04fd593dcd37',
            'version': '0.1.0',
            'name': "SeededGraphMatcher",
            'python_path': 'd3m.primitives.data_transformation.seeded_graph_matcher.ExlineSeededGraphMatcher',
            'source': {
                'name': 'exline',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/cdbethune/d3m-exline/primitives/seeded_graph_matcher.py',
                    'https://github.com/cdbethune/d3m-exline',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/cdbethune/d3m-exline.git@{git_commit}#egg=d3m-exline'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.ARRAY_SLICING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:

        PrimitiveBase.__init__(self, hyperparams=hyperparams, random_seed=random_seed)
        self.unweighted = True
        self.verbose = True
        self.num_iters = 20
        self.tolerance = 1

    def __getstate__(self) -> dict:
        state = PrimitiveBase.__getstate__(self)
        state['models'] = self._model
        return state

    def __setstate__(self, state: dict) -> None:
        PrimitiveBase.__setstate__(self, state)
        self._model = state['models']

    def set_training_data(self, *, inputs: container.DataFrame, outputs: container.DataFrame) -> None:
        self._inputs = inputs
        self._outputs = outputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        logger.debug(f'Fitting {__name__}')
        assert len(self._inputs) == 3

        logger.info(self._inputs)
        
        G1 = self._inputs[1]
        G2 = self._inputs[2]
        logger.info(list(G1.nodes))
        assert isinstance(list(G1.nodes)[0], str)
        assert isinstance(list(G2.nodes)[0], str)
        
        df = self._inputs[0]
        y_train = df['match']
        df.drop(['d3mIndex', 'match'], axis=1, inplace=True)
        logger.info(df.shape)
        assert df.shape[1] == 2

        df.columns = ('orig_id1', 'orig_id2')
        df.orig_id1 = df.orig_id1.astype(str)
        df.orig_id2 = df.orig_id2.astype(str)

        G1, G2, n_nodes = self._pad_graphs(G1, G2)

        G1_nodes = sorted(dict(G1.degree()).items(), key=lambda x: -x[1])
        G1_nodes = list(zip(*G1_nodes))[0]
        G1_lookup = dict(zip(G1.nodes, range(len(G1.nodes))))
        df['num_id1'] = df['orig_id1'].apply(lambda x: G1_lookup[x])

        G2_nodes = sorted(dict(G1.degree()).items(), key=lambda x: -x[1])
        G2_nodes = list(zip(*G2_nodes))[0]
        G2_lookup = dict(zip(G2.nodes, range(len(G2.nodes))))
        df['num_id2'] = df['orig_id2'].apply(lambda x: G2_lookup[x])

        G1p = nx.relabel_nodes(G1, G1_lookup)
        G2p = nx.relabel_nodes(G2, G2_lookup)
        A = nx.adjacency_matrix(G1p, nodelist=list(G1_lookup.values()))
        B = nx.adjacency_matrix(G2p, nodelist=list(G2_lookup.values()))

        # Symmetrize (required by our SGM implementation)
        # Does it hurt performance?
        A = ((A + A.T) > 0).astype(np.float32)
        B = ((B + B.T) > 0).astype(np.float32)

        if self.unweighted:
            A = (A != 0)
            B = (B != 0)
        
        P = df[['num_id1', 'num_id2']][y_train == 1].values
        P = sparse.csr_matrix((np.ones(P.shape[0]), (P[:,0], P[:,1])), shape=(n_nodes, n_nodes))
        
        sgm = ScipyJVClassicSGM(A=A, B=B, P=P, verbose=self.verbose)
        P_out = sgm.run(
            num_iters=self.num_iters,
            tolerance=self.tolerance
        )
        P_out = sparse.csr_matrix((np.ones(n_nodes), (np.arange(n_nodes), P_out)))

        self._model = P_out

        return CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        #train_preds = P[(X_train.num_id1.values, X_train.num_id2.values)]

        # create dataframe to hold d3mIndex and result
        result = self._model.predict(inputs)
        result_df = container.DataFrame({inputs.index.name: inputs.index, self._outputs.columns[0]: result}, generate_metadata=True)

        # mark the semantic types on the dataframe
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        logger.debug(f'\n{result_df}')
        return base.CallResult(result_df)

    def _pad_graphs(self, G1, G2):
        n_nodes = max(G1.order(), G2.order())  
        for i in range(n_nodes - G1.order()):
            G1.add_node('__new_node__salt123_%d' % i)      
        for i in range(n_nodes - G2.order()):
            G2.add_node('__new_node__salt456_%d' % i)     
        assert G1.order() == G2.order()
        return G1, G2, n_nodes

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return