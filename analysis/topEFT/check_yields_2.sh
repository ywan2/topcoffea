curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > conda-install.sh
bash conda-install.sh
y
y
conda create --name myenv python=3.8.3
conda activate myenv
conda install -y -c conda-forge ndcctools conda-pack dill xrootd coffea
