#!/bin/bash


# What we want to call the output files
OUT_FILE_NAME="output_check_yields"

# The json we want to compare against
REF_FILE_NAME="test/ref_yields.json"

# Activate uonda envirnoment
#printf "Activating conda environment..."
unset PYTHONPATH
#eval "$(conda shell.bash hook)"

#conda create myenv

#conda init bash
#source ~/.bashrc    # <- !!!
#conda activate myenv 
#conda activate test-env-01
#source activate test-env-01 

#conda run -n myenv python myenv.py

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

