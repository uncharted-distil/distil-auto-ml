#!/bin/bash

CACHEBUSTER_VAL=0
while getopts ":f" opt; do
  case $opt in
    f)
      CACHEBUSTER_VAL=$(date +%s)
      echo "Forcing re-install of primitives"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

docker build --build-arg CACHEBUSTER=$CACHEBUSTER_VAL -t registry.datadrivendiscovery.org/uncharted/distil-integration/distil-auto-ml:latest .
