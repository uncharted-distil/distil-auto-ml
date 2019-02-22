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
            'data_transformation.imputer.ExlineSimpleImputer = primitives.simple_imputer:SimpleImputerPrimitive',
            'data_transformation.data_cleaning.ExlineReplaceSingletons = primitives.replace_singletons:ReplaceSingletonsPrimitive',
            'data_transformation.imputer.ExlineCategoricalImputer = primitives.categorical_imputer:CategoricalImputerPrimitive',
            'data_transformation.data_cleaning.ExlineEnrichDates = primitives.enrich_dates:EnrichDatesPrimitive',
            'learner.random_forest.ExlineEnsembleForest = primitives.ensemble_forest:EnsembleForestPrimitive',
            'data_transformation.standard_scaler.ExlineStandardScaler = primitives.standard_scalar:StandardScalerPrimitive',
            'data_transformation.one_hot_encoder.ExlineOneHotEncoder = primitives.one_hot_encoder:OneHotEncoderPrimitive',
            'data_transformation.encoder.ExlineBinaryEncoder = primitives.binary_encoder:BinaryEncoderPrimitive'
        ]
    }
)