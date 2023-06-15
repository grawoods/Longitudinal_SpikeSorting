# Longitudinal_SpikeSorting
## Author: Grace Woods, Hong Lab @ Stanford (Spring 2023)
### In Progress! Please check back for updates. 
Questions? Email grawoods@stanford.edu

# Get Intan RHD File Reader
If you want to get Intan loading functions from scratch, use the following:
```
mkdir src
# Get Intan RHD File Reader (wget -P ./src 'link' for Linux/Sherlock, curl -o 'link' for MacOS)

wget -P ./src https://raw.githubusercontent.com/Intan-Technologies/load-rhd-notebook-python/main/importrhdutilities.py

curl -o ./src/importrhdutilities.py https://raw.githubusercontent.com/Intan-Technologies/load-rhd-notebook-python/main/importrhdutilities.py

```
Otherwise, this file is already provided in `src/`.

# If you're working locally, read below:
You really only need the following: 'src/', 'utils.py', and 'sherlock_executable.py'.

An example for running locally is provided below:

'''
python3 sherlock_executable.py --datadir=path/to/data/ --outdir=desired/path/to/results/
'''

Versions of dependencies listed here:
ml python/3.9.0
ml py-numpy/1.20.3_py39
ml py-scikit-learn/1.0.2_py39
ml py-scipy/1.10.1_py39
ml viz
ml py-matplotlib/3.4.2_py39
