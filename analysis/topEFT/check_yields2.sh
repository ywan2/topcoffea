# Activate uonda envirnoment
printf "Activating conda environment..."

export PYTHONPATH=

# Activate the Conda shell hooks without starting a new shell.
CONDA_BASE=$(conda info --base)
. $CONDA_BASE/etc/profile.d/conda.sh

echo "*** Install Conda and Pip packages"

conda update -n base -c defaults conda
conda create -y --name coffea-env
conda activate coffea-env
conda install -y python=3.8.3 six dill
conda install -y -c conda-forge coffea ndcctools conda-pack xrootd uproot

#Install python package
pip install -e .

#echo $?
