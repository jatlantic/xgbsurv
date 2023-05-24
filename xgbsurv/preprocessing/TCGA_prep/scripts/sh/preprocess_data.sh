#!/bin/bash

mkdir -p ./data/processed
mkdir -p ./data/processed/TCGA

Rscript scripts/r/run_preprocessing.R