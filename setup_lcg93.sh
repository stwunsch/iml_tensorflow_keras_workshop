#!/bin/bash

LCG_RELEASE=93
ARCHITECTURE=x86_64-slc6-gcc62-opt

echo "Source LCG release ${LCG_RELEASE} for the architecture ${ARCHITECTURE}."

source /cvmfs/sft.cern.ch/lcg/views/LCG_${LCG_RELEASE}/${ARCHITECTURE}/setup.sh
