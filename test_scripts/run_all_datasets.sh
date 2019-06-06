#!/bin/bash

# Get up to get up
cd ..

# Start TA2
docker-compose up -d distil

# List all the datasets there be in seed_datasets
for dir in seed_datasets_current/*
do
    export DATASET=`python -c "a = '$dir'; print(a.split('/')[1]);"`
    echo $DATASET
    docker-compose up dummy-ta3
    echo $DATASET
done
