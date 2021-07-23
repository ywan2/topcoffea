#!/bin/bash


# What we want to call the output files
OUT_FILE_NAME="output_check_yields"

# The json we want to compare against
REF_FILE_NAME="test/ref_yields.json"

# Activate uonda envirnoment
printf "Activating conda environment..."
unset PYTHONPATH

#export PYTHONPATH=

# Activate the Conda shell hooks without starting a new shell.
#CONDA_BASE=$(conda info --base)
#. $CONDA_BASE/etc/profile.d/conda.sh

#echo "*** Install Conda and Pip packages"

#conda update -n base -c defaults conda
#conda create -y --name coffea-env
#conda activate coffea-env
#conda install -y python=3.8.3 six dill
#conda install -y -c conda-forge coffea ndcctools conda-pack xrootd uproot

#conda run -n coffea-env python run.py ../../topcoffea/cfg/check_yields_sample.cfg -o ${OUT_FILE_NAME}

# Run the processor
printf "Running processor..."
time python run.py ../../topcoffea/cfg/check_yields_sample.cfg -o ${OUT_FILE_NAME}

# Make the jsons
printf "Making yields json from pkl..."
python get_yield_json.py -f histos/${OUT_FILE_NAME}.pkl.gz -n ${OUT_FILE_NAME} --quiet

# If we want this to be the new ref json
#cp ${OUT_FILE_NAME}.json tests/${REF_FILE_NAME}

# Compare the yields to the ref json
printf "Comparing yields agains reference..."
python comp_yields.py ${REF_FILE_NAME} ${OUT_FILE_NAME}.json -t1 "Ref yields" -t2 "New yields" --quiet

# Do something with the exit code?
echo $?

