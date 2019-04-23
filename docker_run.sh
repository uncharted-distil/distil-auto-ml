#!/bin/sh

# $D3MOUTPUTDIR should point to the directory used by the distil server for output
# $D3MINPUTDIR should point to the directory used by the distil server for dataset inpu

docker run \
  --name distil-auto-ml \
  --rm \
  -p 45042:45042 \
  --env D3MOUTPUTDIR=$D3MOUTPUTDIR \
  -v $D3MOUTPUTDIR:$D3MOUTPUTDIR \
  -v $D3MINPUTDIR:$D3MINPUTDIR \
  registry.datadrivendiscovery.org/uncharted_qntfy/ta3ta2_integration/exlineta2:latest
