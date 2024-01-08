#!/usr/bin/env bash

export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
singularity shell --nv --nvccli -B /usr/share/glvnd  build.sif
