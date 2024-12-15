# Bumblebee_Generative
Submission for PHYS 570 Final Project on generative capabilities of Bumblebee

To Reproduce:
1) Download the .yml file and create conda environment via "conda env create -f env.yml" in the directory that the yml file is stored in.
2) Activate the environment using "conda activate env" (or "conda activate <name of environment>")
3) Create an output folder and edit the argument parser in pretraining to link to the directory where your data is stored and your respective output directory.
4) Simply run the PreTrainingV5.py file.

To produce plots:
1) Make sure conda environment is activated.
2) Edit reco_plotting.py and replace BASE_DIR with the the output directory from training and SAVE_DIR to the directory you want to save the plots to.
3) Run reco_plotting.py
