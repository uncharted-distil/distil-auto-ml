"""
   Copyright © 2020 Uncharted Software Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from ast import literal_eval
import unittest
from os import path
import urllib.parse
import numpy as np

from d3m import container
from d3m.metadata import base as metadata_base
from d3m.base import utils as base_utils
from processing.parquet_loader import ParquetDatasetLoader, container_ndarray

_parquet_dataset_path = path.abspath(
    path.join(path.dirname(__file__), "tabular_dataset_parquet")
)
_csv_dataset_path = path.abspath(
    path.join(path.dirname(__file__), "tabular_dataset_csv")
)


class ParquetDatasetLoaderTestCase(unittest.TestCase):
    def test_can_load(self) -> None:
        dataset_path = path.join(_parquet_dataset_path, "datasetDoc.json")
        dataset_url = urllib.parse.urlunparse(("file", "", dataset_path, "", "", ""))
        loader = ParquetDatasetLoader()
        self.assertTrue(loader.can_load(dataset_url))

    def test_cannot_load(self) -> None:
        dataset_path = path.join(_csv_dataset_path, "datasetDoc.json")
        dataset_url = urllib.parse.urlunparse(("file", "", dataset_path, "", "", ""))
        loader = ParquetDatasetLoader()
        self.assertFalse(loader.can_load(dataset_url))

    def test_load(self) -> None:

        dataset_path = path.join(_parquet_dataset_path, "datasetDoc.json")
        dataset_url = urllib.parse.urlunparse(("file", "", dataset_path, "", "", ""))
        loader = ParquetDatasetLoader()
        dataset = loader.load(dataset_uri=dataset_url)

        _, df = base_utils.get_tabular_resource(dataset, "learningData")
        self.assertIsNotNone(df)

        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS))[
                "dimension"
            ]["length"],
            7,
        )

        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 0))[
                "name"
            ],
            "d3mIndex",
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 0))[
                "structural_type"
            ],
            np.int64,
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 0))[
                "semantic_types"
            ],
            (
                "http://schema.org/Integer",
                "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
            ),
        )

        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 1))[
                "name"
            ],
            "alpha",
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 1))[
                "structural_type"
            ],
            np.int64,
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 1))[
                "semantic_types"
            ],
            (
                "http://schema.org/Integer",
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            ),
        )

        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 2))[
                "name"
            ],
            "bravo",
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 2))[
                "structural_type"
            ],
            np.float64,
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 2))[
                "semantic_types"
            ],
            (
                "http://schema.org/Float",
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            ),
        )

        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 3))[
                "name"
            ],
            "charlie",
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 3))[
                "structural_type"
            ],
            np.object_,
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 3))[
                "semantic_types"
            ],
            (
                "http://schema.org/Text",
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            ),
        )

        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 4))[
                "name"
            ],
            "delta",
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 4))[
                "structural_type"
            ],
            np.object_,
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 4))[
                "semantic_types"
            ],
            (
                "https://metadata.datadrivendiscovery.org/types/CategoricalData",
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            ),
        )

        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 5))[
                "name"
            ],
            "echo",
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 5))[
                "structural_type"
            ],
            np.object_,
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 5))[
                "semantic_types"
            ],
            (
                "https://metadata.datadrivendiscovery.org/types/FloatVector",
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            ),
        )

        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 6))[
                "name"
            ],
            "foxtrot",
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 6))[
                "structural_type"
            ],
            np.object_,
        )
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS, 6))[
                "semantic_types"
            ],
            (
                "https://metadata.datadrivendiscovery.org/types/FloatVector",
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            ),
        )

        print(df)

        self.assertEqual(df.iloc[0, 0], 1)
        self.assertEqual(df.iloc[0, 1], 10)
        self.assertEqual(df.iloc[0, 2], 1.0)
        self.assertEqual(df.iloc[0, 3], "xray")
        self.assertEqual(df.iloc[0, 4], " tango sierra")
        self.assertListEqual(df.iloc[0, 5].tolist(), [0, 1, 2, 3])
        self.assertEqual(df.iloc[0, 6], "0,1,2,3")

    def test_lazy_load(self) -> None:
        dataset_path = path.join(_parquet_dataset_path, "datasetDoc.json")
        dataset_url = urllib.parse.urlunparse(("file", "", dataset_path, "", "", ""))
        loader = ParquetDatasetLoader()
        dataset = loader.load(dataset_uri=dataset_url, lazy=True)
        dataset.load_lazy()

        df = base_utils.get_tabular_resource(dataset, "learningData")
        self.assertIsNotNone(df)
        self.assertEqual(
            dataset.metadata.query(("learningData", metadata_base.ALL_ELEMENTS))[
                "dimension"
            ]["length"],
            7,
        )


def csv_to_parquet():
    _csv_dataset_path = path.abspath(
        path.join(path.dirname(__file__), "tabular_dataset_csv")
    )
    dataset_doc_path = path.join(_csv_dataset_path, "datasetDoc.json")

    # load the dataset and convert resource 0 to a dataframe
    dataset = container.Dataset.load(
        "file://{dataset_doc_path}".format(dataset_doc_path=dataset_doc_path)
    )
    df = dataset["learningData"]
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    df.iloc[:, 1] = df.iloc[:, 1].astype(int)
    df.iloc[:, 2] = df.iloc[:, 2].astype(float)
    df.iloc[:, 3] = df.iloc[:, 3].astype(str)
    df.iloc[:, 4] = df.iloc[:, 4].astype(str)
    df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: literal_eval(f"[{x}]"))
    df.iloc[:, 6] = df.iloc[:, 6].astype(str)

    df.to_parquet(
        path.join(_parquet_dataset_path, "tables", "learningData.parquet"), index=False
    )


if __name__ == "__main__":
    unittest.main()
