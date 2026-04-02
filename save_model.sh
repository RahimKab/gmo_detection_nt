#!/bin/bash

if command -v snakemake >/dev/null 2>&1; then
    snakemake --snakefile /home/strange/Documents/master_2/internship/model/Snakefile --cores 1 save_model
else
    /home/strange/Documents/master_2/internship/model/.venv/bin/python /home/strange/Documents/master_2/internship/model/scripts/utils/save_model.py
fi
