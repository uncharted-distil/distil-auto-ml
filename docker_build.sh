#!/bin/bash

CACHEBUSTER_VAL=0
while getopts ":fg" opt; do
  case $opt in
    f)
      CACHEBUSTER_VAL=$(date +%s)
      echo "Forcing re-install of primitives"
      ;;
    g)
      GPU_MXNET="[gpu-cuda-10.1]"
      echo "Using MXNET version $GPU_MXNET"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

docker build --build-arg CACHEBUSTER=$CACHEBUSTER_VAL --build-arg GPU_MXNET=$GPU_MXNET -t registry.datadrivendiscovery.org/uncharted/distil-integration/distil-auto-ml:latest .
