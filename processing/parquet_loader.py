import os
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple
from urllib import parse as url_parse
import numpy as np
import pandas as pd
from d3m.container import pandas as container_pandas, ndarray as container_ndarray
from d3m.metadata import base as metadata_base
from d3m.container.dataset import (
    D3MDatasetLoader,
    ComputeDigest,
    Dataset,
    D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES,
    D3M_ROLE_CONSTANTS_TO_SEMANTIC_TYPES,
)
from d3m import exceptions
from pandas.core.arrays.sparse import dtype

logger = logging.getLogger(__name__)


class ParquetDatasetLoader(D3MDatasetLoader):
    """
    A class for loading a dataset from a parquet file.  Currently relies on the presence of a datasetDoc for
    type info.

    Loader supports both loading a dataset from a local file system or remote locations.
    URI should point to a file with ``.parquet`` file extension.
    """

    # checks to see if this is a simple d3m tabular dataset that stores the table data
    # as a parquet file
    def can_load(self, dataset_uri: str) -> bool:
        # check the base can_load to make sure we're pointing to a valid datasetDoc.json
        can_load = super().can_load(dataset_uri)
        if not can_load:
            return False

        # check to see that there is a parquet file for the learning data
        dataset_doc = self._load_dataset_doc(dataset_uri)
        if dataset_doc is None:
            return False

        return self._validate_dataset_doc(dataset_doc)

    def load(
        self,
        dataset_uri: str,
        *,
        dataset_id: str = None,
        dataset_version: str = None,
        dataset_name: str = None,
        lazy: bool = False,
        compute_digest: ComputeDigest = ComputeDigest.ONLY_IF_MISSING,
        strict_digest: bool = False,
        handle_score_split: bool = True,
    ) -> "Dataset":
        parsed_uri = url_parse.urlparse(dataset_uri, allow_fragments=False)

        # Pandas requires a host for "file" URIs.
        if parsed_uri.scheme == "file" and parsed_uri.netloc == "":
            parsed_uri = parsed_uri._replace(netloc="localhost")
            dataset_uri = url_parse.urlunparse(parsed_uri)

        dataset_size = None

        resources: Dict = {}
        metadata = metadata_base.DataMetadata()

        if not lazy:
            load_lazy = None
            metadata, resources = self._load_data(metadata, dataset_uri=dataset_uri)
        else:

            def load_lazy(dataset: Dataset) -> None:
                dataset.metadata, resources = self._load_data(
                    dataset.metadata, dataset_uri=dataset_uri
                )
                for k, v in resources.items():
                    dataset[k] = v

                new_metadata = {
                    "dimension": {"length": len(dataset)},
                    "stored_size": dataset_size,
                }

                dataset.metadata = dataset.metadata.update((), new_metadata)
                dataset._load_lazy = None

        if dataset_id is None:
            dataset_doc = self._load_dataset_doc(dataset_uri)
            if dataset_doc is None:
                raise exceptions.InvalidDatasetError(
                    f"Dataset '{dataset_uri}' can't be found."
                )
            dataset_id = dataset_doc["about"]["datasetID"]

        dataset_metadata = {
            "schema": metadata_base.CONTAINER_SCHEMA_VERSION,
            "structural_type": Dataset,
            "id": dataset_id,
            "digest": "D3ADB33F",  # how to compute proper digest from parquet file
            "name": dataset_name or os.path.basename(parsed_uri.path),
            "location_uris": [
                dataset_uri,
            ],
            "dimension": {
                "name": "resources",
                "semantic_types": [
                    "https://metadata.datadrivendiscovery.org/types/DatasetResource"
                ],
                "length": len(resources),
            },
        }

        if dataset_version is not None:
            dataset_metadata["version"] = dataset_version

        if dataset_size is not None:
            dataset_metadata["stored_size"] = dataset_size

        metadata = metadata.update((), dataset_metadata)

        return Dataset(resources, metadata, load_lazy=load_lazy)

    def _load_dataset_doc(self, dataset_uri: str) -> Optional[Dict[Any, Any]]:
        # load the metadata to see if we're pointing to a parquet file
        parsed_url = url_parse.urlparse(dataset_uri)
        dataset_doc_path = parsed_url.path
        try:
            with open(dataset_doc_path, "r", encoding="utf8") as dataset_doc_file:
                return json.load(dataset_doc_file)
        except FileNotFoundError:
            return None

    # returns true if the referenced dataset is a single table d3m dataset where the table
    # data is stored as a parquet file
    def _validate_dataset_doc(self, dataset_doc) -> bool:
        # this should be a dataset consisting of a single resource
        if len(dataset_doc["dataResources"]) is not 1:
            return False

        mainResource = dataset_doc["dataResources"][0]

        # should be a table resource
        resType = mainResource["resType"]
        if resType != "table":
            return False

        # table should be a parquet file
        resFormat = mainResource["resFormat"]
        if next(iter(resFormat)) != "application/parquet":
            return False

        return True

    def _load_data(
        self, metadata, *, dataset_uri: str
    ) -> Tuple[metadata_base.DataMetadata, Dict]:
        dataset_doc = self._load_dataset_doc(dataset_uri)
        if dataset_doc is None:
            raise exceptions.InvalidDatasetError(
                f"Dataset '{dataset_uri}' can't be found."
            )

        # validate the dataset doc
        if not self._validate_dataset_doc(dataset_doc):
            raise exceptions.InvalidDatasetError(
                f"Dataset '{dataset_uri}' is not a valid parquet dataset."
            )

        # find the resource path
        resources = dataset_doc["dataResources"][0]
        resource_path = resources["resPath"]

        # read in the parquet data
        parsed_url = url_parse.urlparse(dataset_uri)
        tables_path = os.path.join(os.path.dirname(parsed_url.path), resource_path)
        raw_df = pd.read_parquet(tables_path)

        df = container_pandas.DataFrame(raw_df)

        resource_id = resources["resID"]
        resources_loaded = {}
        resources_loaded[resource_id] = df

        if "d3mIndex" not in df.columns:
            df.insert(0, "d3mIndex", range(len(df)))
            d3m_index_generated = True
        else:
            d3m_index_generated = False

        metadata = metadata.update(
            (resource_id,),
            {
                "structural_type": type(df),
                "semantic_types": [
                    "https://metadata.datadrivendiscovery.org/types/Table",
                    "https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint",
                ],
                "dimension": {
                    "name": "rows",
                    "semantic_types": [
                        "https://metadata.datadrivendiscovery.org/types/TabularRow"
                    ],
                    "length": len(df),
                },
            },
        )

        metadata = metadata.update(
            ("learningData", metadata_base.ALL_ELEMENTS),
            {
                "dimension": {
                    "name": "columns",
                    "semantic_types": [
                        "https://metadata.datadrivendiscovery.org/types/TabularColumn"
                    ],
                    "length": len(df.columns),
                },
            },
        )

        dataset_doc_columns = resources["columns"]
        columns_by_index = {
            int(col_data["colIndex"]): col_data for col_data in dataset_doc_columns
        }
        for i, column_name in enumerate(df.columns):
            if i == 0 and d3m_index_generated:
                metadata = metadata.update(
                    (resource_id, metadata_base.ALL_ELEMENTS, i),
                    {
                        "name": column_name,
                        "structural_type": np.int64,
                        "semantic_types": [
                            "http://schema.org/Integer",
                            "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
                        ],
                    },
                )
            else:
                # populate semantic types from column role and type
                col_data = columns_by_index[i]
                semantic_types = [
                    D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES[col_data["colType"]]
                ]
                semantic_types += [
                    D3M_ROLE_CONSTANTS_TO_SEMANTIC_TYPES[role]
                    for role in col_data["role"]
                ]
                structural_type = df.dtypes[i].type
                metadata = metadata.update(
                    ("learningData", metadata_base.ALL_ELEMENTS, i),
                    {
                        "name": column_name,
                        "structural_type": structural_type,
                        "semantic_types": semantic_types,
                    },
                )

                # remove any possible surrounding characters from a string representation of a float vector
                if (
                    "https://metadata.datadrivendiscovery.org/types/FloatVector"
                    in semantic_types
                    and df.dtypes[column_name] == "object"
                ):
                    df[column_name] = df[column_name].replace(
                        to_replace='[\[\]()<>""{}]', value="", regex=True
                    )

        return metadata, resources_loaded
