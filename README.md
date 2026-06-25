# IMEGN
Interpretable Ligand-Protein Affinity Prediction Framework using Multi-Expert Spatial Graph Networks and Large Language Models

This paper includes accompanying code.

train.py contains the code for training our model.

If you encounter missing information while reproducing our code, please download the original PDB data from the link first, as the original data is too large to upload to GitHub.

# Datasets

All data used in this paper are publicly available and can be accessed here: This contains the files "test_2016", "test_hiq", "test", and "test_2013".

PDBbind v2020: http://www.pdbbind.org.cn/download.php

2013 and 2016 core sets: http://www.pdbbind.org.cn/casf.php

Before using the code, please use preprocess_3d.py and update_pocket_geometry.py to obtain and update the target geometric features.
