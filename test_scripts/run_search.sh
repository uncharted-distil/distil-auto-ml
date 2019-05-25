#!/bin/bash

DATASET=$1

[ -z "$D3MINPUTDIR" ] && echo "Error - D3MINPUTDIR unset" && exit 1
[ -z "$D3MOUTPUTDIR" ] && echo "Error - D3MOUTPUTDIR unset" && exit 1
[ -z "$DATASET" ] && echo "Error - DATASET unset" && exit 1

python3 -m dummy_ta3.dummy_ta3 \
    -p $D3MINPUTDIR/$DATASET/${DATASET}_problem/problemDoc.json \
    -d $D3MINPUTDIR \
    -e localhost \
    -t 300
