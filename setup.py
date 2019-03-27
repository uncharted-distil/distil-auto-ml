#!/usr/bin/env/python

from setuptools import setup, find_packages

setup(
    name='exline',
    author='Ben Johnson',
    author_email='bkj.322@gmail.com',
    classifiers=[],
    description='exline',
    keywords=['exline'],
    license='ALV2',
    packages=find_packages(),
    version="0.0.0",
    entry_points={
        'd3m.primitives': [
            'data_transformation.imputer.ExlineSimpleImputer = exline.primitives.simple_imputer:SimpleImputerPrimitive',
            'data_transformation.data_cleaning.ExlineReplaceSingletons = exline.primitives.replace_singletons:ReplaceSingletonsPrimitive',
            'data_transformation.imputer.ExlineCategoricalImputer = exline.primitives.categorical_imputer:CategoricalImputerPrimitive',
            'data_transformation.data_cleaning.ExlineEnrichDates = exline.primitives.enrich_dates:EnrichDatesPrimitive',
            'learner.random_forest.ExlineEnsembleForest = exline.primitives.ensemble_forest:EnsembleForestPrimitive',
            'data_transformation.standard_scaler.ExlineStandardScaler = exline.primitives.standard_scaler:StandardScalerPrimitive',
            'data_transformation.one_hot_encoder.ExlineOneHotEncoder = exline.primitives.one_hot_encoder:OneHotEncoderPrimitive',
            'data_transformation.encoder.ExlineBinaryEncoder = exline.primitives.binary_encoder:BinaryEncoderPrimitive',
            'data_transformation.column_parser.ExlineSimpleColumnParser = exline.primitives.simple_column_parser:SimpleColumnParserPrimitive',
            'data_transformation.missing_indicator.ExlineMissingIndicator = exline.primitives.missing_indicator:MissingIndicatorPrimitive',
            'data_transformation.data_cleaning.ExlineZeroColumnRemover = exline.primitives.zero_column_remover:ZeroColumnRemoverPrimitive'
        ]
    }
)