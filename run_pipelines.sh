#!/bin/bash

ulimit -n 65536

for f in pipelines/*.sh; do
  bash "$f"
done