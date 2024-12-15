import sys
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import awkward as ak
import uproot
from scipy.stats import wasserstein_distance
import vector
from matplotlib.backends.backend_pdf import PdfPages

# *** very important parameters *** edit at your own risk lol
get_data_from_file = False
post_as_pngs = True # alternative is pdf
N_BINS = 40
BASE_DIR = r'/depot/cms/top/jprodger/Bumblebee/src/Experiment121324/output/'
SAVE_DIR = BASE_DIR
#SAVE_DIR = r'/depot/cms/top/jprodger/Bumblebee/src/reco_plot_generation/'

if get_data_from_file:
    
    file = uproot.open(SAVE_DIR + 'bumblebee_reco.root')
    tree = file["Bumblebee"]
    branches = tree.arrays()
    
    gen_l_pt = ak.to_numpy(branches['gen_l_pt'])
    gen_lbar_pt = ak.to_numpy(branches['gen_lbar_pt'])
    gen_b_pt = ak.to_numpy(branches['gen_b_pt'])
    gen_bbar_pt = ak.to_numpy(branches['gen_bbar_pt'])
    gen_n_pt = ak.to_numpy(branches['gen_n_pt'])
    gen_nbar_pt = ak.to_numpy(branches['gen_nbar_pt'])
    gen_l_eta = ak.to_numpy(branches['gen_l_eta'])
    gen_lbar_eta = ak.to_numpy(branches['gen_lbar_eta'])
    gen_b_eta = ak.to_numpy(branches['gen_b_eta'])
    gen_bbar_eta = ak.to_numpy(branches['gen_bbar_eta'])
    gen_n_eta = ak.to_numpy(branches['gen_n_eta'])
    gen_nbar_eta = ak.to_numpy(branches['gen_nbar_eta'])
    gen_l_phi = ak.to_numpy(branches['gen_l_phi'])
    gen_lbar_phi = ak.to_numpy(branches['gen_lbar_phi'])
    gen_b_phi = ak.to_numpy(branches['gen_b_phi'])
    gen_bbar_phi = ak.to_numpy(branches['gen_bbar_phi'])
    gen_n_phi = ak.to_numpy(branches['gen_n_phi'])
    gen_nbar_phi = ak.to_numpy(branches['gen_nbar_phi'])
    gen_l_mass = ak.to_numpy(branches['gen_l_mass'])
    gen_lbar_mass = ak.to_numpy(branches['gen_lbar_mass'])
    gen_b_mass = ak.to_numpy(branches['gen_b_mass'])
    gen_bbar_mass = ak.to_numpy(branches['gen_bbar_mass'])
    gen_n_mass = ak.to_numpy(branches['gen_n_mass'])
    gen_nbar_mass = ak.to_numpy(branches['gen_nbar_mass'])
    gen_met_pt = ak.to_numpy(branches['gen_met_pt'])
    gen_met_phi = ak.to_numpy(branches['gen_met_phi'])
    gen_met_eta = ak.to_numpy(branches['gen_met_eta'])
    gen_met_mass = ak.to_numpy(branches['gen_met_mass'])
    
    pred_l_pt = ak.to_numpy(branches['pred_l_pt'])
    pred_lbar_pt = ak.to_numpy(branches['pred_lbar_pt'])
    pred_b_pt = ak.to_numpy(branches['pred_b_pt'])
    pred_bbar_pt = ak.to_numpy(branches['pred_bbar_pt'])
    pred_n_pt = ak.to_numpy(branches['pred_n_pt'])
    pred_nbar_pt = ak.to_numpy(branches['pred_nbar_pt'])
    pred_l_eta = ak.to_numpy(branches['pred_l_eta'])
    pred_lbar_eta = ak.to_numpy(branches['pred_lbar_eta'])
    pred_b_eta = ak.to_numpy(branches['pred_b_eta'])
    pred_bbar_eta = ak.to_numpy(branches['pred_bbar_eta'])
    pred_n_eta = ak.to_numpy(branches['pred_n_eta'])
    pred_nbar_eta = ak.to_numpy(branches['pred_nbar_eta'])
    pred_l_phi = ak.to_numpy(branches['pred_l_phi'])
    pred_lbar_phi = ak.to_numpy(branches['pred_lbar_phi'])
    pred_b_phi = ak.to_numpy(branches['pred_b_phi'])
    pred_bbar_phi = ak.to_numpy(branches['pred_bbar_phi'])
    pred_n_phi = ak.to_numpy(branches['pred_n_phi'])
    pred_nbar_phi = ak.to_numpy(branches['pred_nbar_phi'])
    pred_l_mass = ak.to_numpy(branches['pred_l_mass'])
    pred_lbar_mass = ak.to_numpy(branches['pred_lbar_mass'])
    pred_b_mass = ak.to_numpy(branches['pred_b_mass'])
    pred_bbar_mass = ak.to_numpy(branches['pred_bbar_mass'])
    pred_n_mass = ak.to_numpy(branches['pred_n_mass'])
    pred_nbar_mass = ak.to_numpy(branches['pred_nbar_mass'])
    pred_met_pt = ak.to_numpy(branches['pred_met_pt'])
    pred_met_phi = ak.to_numpy(branches['pred_met_phi'])
    pred_met_eta = ak.to_numpy(branches['pred_met_eta'])
    pred_met_mass = ak.to_numpy(branches['pred_met_mass'])
    
    pred_t_mass = ak.to_numpy(branches['pred_t_mass'])
    pred_t_pt = ak.to_numpy(branches['pred_t_pt'])
    pred_t_eta = ak.to_numpy(branches['pred_t_eta'])
    pred_t_phi = ak.to_numpy(branches['pred_t_phi'])
    pred_t_px = ak.to_numpy(branches['pred_t_px'])
    pred_t_py = ak.to_numpy(branches['pred_t_py'])
    pred_t_pz = ak.to_numpy(branches['pred_t_pz'])
    pred_tbar_mass = ak.to_numpy(branches['pred_tbar_mass'])
    pred_tbar_pt = ak.to_numpy(branches['pred_tbar_pt'])
    pred_tbar_eta = ak.to_numpy(branches['pred_tbar_eta'])
    pred_tbar_phi = ak.to_numpy(branches['pred_tbar_phi'])
    pred_tbar_px = ak.to_numpy(branches['pred_tbar_px'])
    pred_tbar_py = ak.to_numpy(branches['pred_tbar_py'])
    pred_tbar_pz = ak.to_numpy(branches['pred_tbar_pz'])
    
    
    gen_t_mass = ak.to_numpy(branches['gen_t_mass'])
    gen_t_pt = ak.to_numpy(branches['gen_t_pt'])
    gen_t_eta = ak.to_numpy(branches['gen_t_eta'])
    gen_t_phi = ak.to_numpy(branches['gen_t_phi'])
    gen_t_px = ak.to_numpy(branches['gen_t_px'])
    gen_t_py = ak.to_numpy(branches['gen_t_py'])
    gen_t_pz = ak.to_numpy(branches['gen_t_pz'])
    gen_tbar_mass = ak.to_numpy(branches['gen_tbar_mass'])
    gen_tbar_pt = ak.to_numpy(branches['gen_tbar_pt'])
    gen_tbar_eta = ak.to_numpy(branches['gen_tbar_eta'])
    gen_tbar_phi = ak.to_numpy(branches['gen_tbar_phi'])
    gen_tbar_px = ak.to_numpy(branches['gen_tbar_px'])
    gen_tbar_py = ak.to_numpy(branches['gen_tbar_py'])
    gen_tbar_pz = ak.to_numpy(branches['gen_tbar_pz'])
    
    pred_Wp_mass = ak.to_numpy(branches['pred_Wp_mass'])
    pred_Wp_pt = ak.to_numpy(branches['pred_Wp_pt'])
    pred_Wp_eta = ak.to_numpy(branches['pred_Wp_eta'])
    pred_Wp_phi = ak.to_numpy(branches['pred_Wp_phi'])
    pred_Wp_px = ak.to_numpy(branches['pred_Wp_px'])
    pred_Wp_py = ak.to_numpy(branches['pred_Wp_py'])
    pred_Wp_pz = ak.to_numpy(branches['pred_Wp_pz'])
    pred_Wm_mass = ak.to_numpy(branches['pred_Wm_mass'])
    pred_Wm_pt = ak.to_numpy(branches['pred_Wm_pt'])
    pred_Wm_eta = ak.to_numpy(branches['pred_Wm_eta'])
    pred_Wm_phi = ak.to_numpy(branches['pred_Wm_phi'])
    pred_Wm_px = ak.to_numpy(branches['pred_Wm_px'])
    pred_Wm_py = ak.to_numpy(branches['pred_Wm_py'])
    pred_Wm_pz = ak.to_numpy(branches['pred_Wm_pz'])
    
    
    gen_Wp_mass = ak.to_numpy(branches['gen_Wp_mass'])
    gen_Wp_pt = ak.to_numpy(branches['gen_Wp_pt'])
    gen_Wp_eta = ak.to_numpy(branches['gen_Wp_eta'])
    gen_Wp_phi = ak.to_numpy(branches['gen_Wp_phi'])
    gen_Wp_px = ak.to_numpy(branches['gen_Wp_px'])
    gen_Wp_py = ak.to_numpy(branches['gen_Wp_py'])
    gen_Wp_pz = ak.to_numpy(branches['gen_Wp_pz'])
    gen_Wm_mass = ak.to_numpy(branches['gen_Wm_mass'])
    gen_Wm_pt = ak.to_numpy(branches['gen_Wm_pt'])
    gen_Wm_eta = ak.to_numpy(branches['gen_Wm_eta'])
    gen_Wm_phi = ak.to_numpy(branches['gen_Wm_phi'])
    gen_Wm_px = ak.to_numpy(branches['gen_Wm_px'])
    gen_Wm_py = ak.to_numpy(branches['gen_Wm_py'])
    gen_Wm_pz = ak.to_numpy(branches['gen_Wm_pz'])
    
    reco_l_pt = ak.to_numpy(branches['reco_l_pt'])
    reco_lbar_pt = ak.to_numpy(branches['reco_lbar_pt'])
    reco_b_pt = ak.to_numpy(branches['reco_b_pt'])
    reco_bbar_pt = ak.to_numpy(branches['reco_bbar_pt'])
    reco_met_pt = ak.to_numpy(branches['reco_met_pt'])
    reco_l_phi = ak.to_numpy(branches['reco_l_phi'])
    reco_lbar_phi = ak.to_numpy(branches['reco_lbar_phi'])
    reco_b_phi = ak.to_numpy(branches['reco_b_phi'])
    reco_bbar_phi = ak.to_numpy(branches['reco_bbar_phi'])
    reco_met_phi = ak.to_numpy(branches['reco_met_phi'])
    reco_l_eta = ak.to_numpy(branches['reco_l_eta'])
    reco_lbar_eta = ak.to_numpy(branches['reco_lbar_eta'])
    reco_b_eta = ak.to_numpy(branches['reco_b_eta'])
    reco_bbar_eta = ak.to_numpy(branches['reco_bbar_eta'])
    reco_met_eta = ak.to_numpy(branches['reco_met_eta'])
    reco_l_mass = ak.to_numpy(branches['reco_l_mass'])
    reco_lbar_mass = ak.to_numpy(branches['reco_lbar_mass'])
    reco_b_mass = ak.to_numpy(branches['reco_b_mass'])
    reco_bbar_mass = ak.to_numpy(branches['reco_bbar_mass'])
    reco_met_mass = ak.to_numpy(branches['reco_met_mass'])
    reco_t_pt = ak.to_numpy(branches['reco_t_pt'])
    reco_t_eta = ak.to_numpy(branches['reco_t_eta'])
    reco_t_phi = ak.to_numpy(branches['reco_t_phi'])
    reco_t_mass = ak.to_numpy(branches['reco_t_mass'])
    reco_t_px = ak.to_numpy(branches['reco_t_px'])
    reco_t_py = ak.to_numpy(branches['reco_t_py'])
    reco_t_pz = ak.to_numpy(branches['reco_t_pz'])
    reco_tbar_pt = ak.to_numpy(branches['reco_tbar_pt'])
    reco_tbar_eta = ak.to_numpy(branches['reco_tbar_eta'])
    reco_tbar_phi = ak.to_numpy(branches['reco_tbar_phi'])
    reco_tbar_mass = ak.to_numpy(branches['reco_tbar_mass'])
    reco_tbar_px = ak.to_numpy(branches['reco_tbar_px'])
    reco_tbar_py = ak.to_numpy(branches['reco_tbar_py'])
    reco_tbar_pz = ak.to_numpy(branches['reco_tbar_pz'])
    reco_ttbar_pt = ak.to_numpy(branches['reco_ttbar_pt'])
    reco_ttbar_eta = ak.to_numpy(branches['reco_ttbar_eta'])
    reco_ttbar_phi = ak.to_numpy(branches['reco_ttbar_phi'])
    reco_ttbar_mass = ak.to_numpy(branches['reco_ttbar_mass'])
    pred_ttbar_pt = ak.to_numpy(branches['pred_ttbar_pt'])
    pred_ttbar_eta = ak.to_numpy(branches['pred_ttbar_eta'])
    pred_ttbar_phi = ak.to_numpy(branches['pred_ttbar_phi'])
    pred_ttbar_mass = ak.to_numpy(branches['pred_ttbar_mass'])
    gen_ttbar_pt = ak.to_numpy(branches['gen_ttbar_pt'])
    gen_ttbar_eta = ak.to_numpy(branches['gen_ttbar_eta'])
    gen_ttbar_phi = ak.to_numpy(branches['gen_ttbar_phi'])
    gen_ttbar_mass = ak.to_numpy(branches['gen_ttbar_mass'])
    

else:
    gen_means = torch.load(BASE_DIR + 'test_gen_means.pt')
    reco_means = torch.load(BASE_DIR + 'test_reco_means.pt')
    reco_stdevs = torch.load(BASE_DIR + 'test_reco_stdevs.pt')
    gen_stdevs = torch.load(BASE_DIR + 'test_gen_stdevs.pt')
    
    truths = torch.load(BASE_DIR + 'generative_truths.pt')
    predictions = torch.load(BASE_DIR + 'generative_predictions.pt')
    
    gen_l_pt = []
    gen_lbar_pt = []
    gen_b_pt = []
    gen_bbar_pt = []
    gen_n_pt = []
    gen_nbar_pt = []
    gen_l_eta = []
    gen_lbar_eta = []
    gen_b_eta = []
    gen_bbar_eta = []
    gen_n_eta = []
    gen_nbar_eta = []
    gen_l_phi = []
    gen_lbar_phi = []
    gen_b_phi = []
    gen_bbar_phi = []
    gen_n_phi = []
    gen_nbar_phi = []
    gen_l_mass = []
    gen_lbar_mass = []
    gen_b_mass = []
    gen_bbar_mass = []
    gen_n_mass = []
    gen_nbar_mass = []
    gen_met_pt = []
    gen_met_phi = []
    gen_met_eta = []
    gen_met_mass = []
    
    pred_l_pt = []
    pred_lbar_pt = []
    pred_b_pt = []
    pred_bbar_pt = []
    pred_l_eta = []
    pred_lbar_eta = []
    pred_b_eta = []
    pred_bbar_eta = []
    pred_l_phi = []
    pred_lbar_phi = []
    pred_b_phi = []
    pred_bbar_phi = []
    pred_l_mass = []
    pred_lbar_mass = []
    pred_b_mass = []
    pred_bbar_mass = []
    pred_met_pt = []
    pred_met_phi = []
    pred_met_mass = []
    pred_met_eta = []
    
    
    reco_l_pt = []
    reco_lbar_pt = []
    reco_b_pt = []
    reco_bbar_pt = []
    reco_met_pt = []
    reco_l_eta = []
    reco_lbar_eta = []
    reco_b_eta = []
    reco_bbar_eta = []
    reco_met_eta = []
    reco_l_phi = []
    reco_lbar_phi = []
    reco_b_phi = []
    reco_bbar_phi = []
    reco_met_phi = []
    reco_l_mass = []
    reco_lbar_mass = []
    reco_b_mass = []
    reco_bbar_mass = []
    reco_met_mass = []
    
    
    for i in tqdm(range(len(predictions))):
        for j in range(len(predictions[i])):
            # gen pt
            gen_l_pt.append((np.exp(truths[i][j][5][0]) * gen_stdevs[0][0]) + gen_means[0][0])
            gen_lbar_pt.append((np.exp(truths[i][j][6][0]) * gen_stdevs[1][0]) + gen_means[1][0])
            gen_b_pt.append((np.exp(truths[i][j][7][0]) * gen_stdevs[2][0]) + gen_means[2][0])
            gen_bbar_pt.append((np.exp(truths[i][j][8][0]) * gen_stdevs[3][0]) + gen_means[3][0])
            gen_n_pt.append((np.exp(truths[i][j][9][0]) * gen_stdevs[4][0]) + gen_means[4][0])
            gen_nbar_pt.append((np.exp(truths[i][j][10][0]) * gen_stdevs[5][0]) + gen_means[5][0])
    
            # gen eta
            gen_l_eta.append((truths[i][j][5][1] * gen_stdevs[0][1]) + gen_means[0][1])
            gen_lbar_eta.append((truths[i][j][6][1] * gen_stdevs[1][1]) + gen_means[1][1])
            gen_b_eta.append((truths[i][j][7][1] * gen_stdevs[2][1]) + gen_means[2][1])
            gen_bbar_eta.append((truths[i][j][8][1] * gen_stdevs[3][1]) + gen_means[3][1])
            gen_n_eta.append((truths[i][j][9][1] * gen_stdevs[4][1]) + gen_means[4][1])
            gen_nbar_eta.append((truths[i][j][10][1] * gen_stdevs[5][1]) + gen_means[5][1])
    
            # gen phi
            gen_l_phi.append((truths[i][j][5][2] * gen_stdevs[0][2]) + gen_means[0][2])
            gen_lbar_phi.append((truths[i][j][6][2] * gen_stdevs[1][2]) + gen_means[1][2])
            gen_b_phi.append((truths[i][j][7][2] * gen_stdevs[2][2]) + gen_means[2][2])
            gen_bbar_phi.append((truths[i][j][8][2] * gen_stdevs[3][2]) + gen_means[3][2])
            gen_n_phi.append((truths[i][j][9][2] * gen_stdevs[4][2]) + gen_means[4][2])
            gen_nbar_phi.append((truths[i][j][10][2] * gen_stdevs[5][2]) + gen_means[5][2])
    
            # gen mass
            gen_l_mass.append((truths[i][j][5][3] * gen_stdevs[0][3]) + gen_means[0][3])
            gen_lbar_mass.append((truths[i][j][6][3] * gen_stdevs[1][3]) + gen_means[1][3])
            gen_b_mass.append((truths[i][j][7][3] * gen_stdevs[2][3]) + gen_means[2][3])
            gen_bbar_mass.append((truths[i][j][8][3] * gen_stdevs[3][3]) + gen_means[3][3])
            gen_n_mass.append((truths[i][j][9][3] * gen_stdevs[4][3]) + gen_means[4][3])
            gen_nbar_mass.append((truths[i][j][10][3] * gen_stdevs[5][3]) + gen_means[5][3])
    
            # pred pt
            pred_l_pt.append((np.exp(predictions[i][j][0][0]) * gen_stdevs[0][0]) + gen_means[0][0])
            pred_lbar_pt.append((np.exp(predictions[i][j][1][0]) * gen_stdevs[1][0]) + gen_means[1][0])
            pred_b_pt.append((np.exp(predictions[i][j][2][0]) * gen_stdevs[2][0]) + gen_means[2][0])
            pred_bbar_pt.append((np.exp(predictions[i][j][3][0]) * gen_stdevs[3][0]) + gen_means[3][0])
            pred_met_pt.append((np.exp(predictions[i][j][4][0]) * gen_stdevs[4][0]) + gen_means[4][0])
    
            # pred eta
            pred_l_eta.append((predictions[i][j][0][1] * gen_stdevs[0][1]) + gen_means[0][1])
            pred_lbar_eta.append((predictions[i][j][1][1] * gen_stdevs[1][1]) + gen_means[1][1])
            pred_b_eta.append((predictions[i][j][2][1] * gen_stdevs[2][1]) + gen_means[2][1])
            pred_bbar_eta.append((predictions[i][j][3][1] * gen_stdevs[3][1]) + gen_means[3][1])
            pred_met_eta.append((predictions[i][j][4][1] * gen_stdevs[4][1]) + gen_means[4][1])
    
            # pred phi
            pred_l_phi.append((predictions[i][j][0][2] * gen_stdevs[0][2]) + gen_means[0][2])
            pred_lbar_phi.append((predictions[i][j][1][2] * gen_stdevs[1][2]) + gen_means[1][2])
            pred_b_phi.append((predictions[i][j][2][2] * gen_stdevs[2][2]) + gen_means[2][2])
            pred_bbar_phi.append((predictions[i][j][3][2] * gen_stdevs[3][2]) + gen_means[3][2])
            pred_met_phi.append((predictions[i][j][4][2] * gen_stdevs[4][2]) + gen_means[4][2])
    
            # pred mass
            pred_l_mass.append((predictions[i][j][0][3] * gen_stdevs[0][3]) + gen_means[0][3])
            pred_lbar_mass.append((predictions[i][j][1][3] * gen_stdevs[1][3]) + gen_means[1][3])
            pred_b_mass.append((predictions[i][j][2][3] * gen_stdevs[2][3]) + gen_means[2][3])
            pred_bbar_mass.append((predictions[i][j][3][3] * gen_stdevs[3][3]) + gen_means[3][3])
            pred_met_mass.append((predictions[i][j][4][3] * gen_stdevs[4][3]) + gen_means[4][3])
            # TODO: fix pred stuff
            
            # reco pt
            reco_l_pt.append((np.exp(truths[i][j][0][0]) * reco_stdevs[0][0]) + reco_means[0][0])
            reco_lbar_pt.append((np.exp(truths[i][j][1][0]) * reco_stdevs[1][0]) + reco_means[1][0])
            reco_b_pt.append((np.exp(truths[i][j][2][0]) * reco_stdevs[2][0]) + reco_means[2][0])
            reco_bbar_pt.append((np.exp(truths[i][j][3][0]) * reco_stdevs[3][0]) + reco_means[3][0])
            reco_met_pt.append((np.exp(truths[i][j][4][0]) * reco_stdevs[4][0]) + reco_means[4][0])
    
            # reco eta
            reco_l_eta.append((truths[i][j][0][1] * reco_stdevs[0][1]) + reco_means[0][1])
            reco_lbar_eta.append((truths[i][j][1][1] * reco_stdevs[1][1]) + reco_means[1][1])
            reco_b_eta.append((truths[i][j][2][1] * reco_stdevs[2][1]) + reco_means[2][1])
            reco_bbar_eta.append((truths[i][j][3][1] * reco_stdevs[3][1]) + reco_means[3][1])
            reco_met_eta.append((truths[i][j][4][1] * reco_stdevs[4][1]) + reco_means[4][1])
    
            # reco phi
            reco_l_phi.append((truths[i][j][0][2] * reco_stdevs[0][2]) + reco_means[0][2])
            reco_lbar_phi.append((truths[i][j][1][2] * reco_stdevs[1][2]) + reco_means[1][2])
            reco_b_phi.append((truths[i][j][2][2] * reco_stdevs[2][2]) + reco_means[2][2])
            reco_bbar_phi.append((truths[i][j][3][2] * reco_stdevs[3][2]) + reco_means[3][2])
            reco_met_phi.append((truths[i][j][4][2] * reco_stdevs[4][2]) + reco_means[4][2])
    
            # reco mass
            reco_l_mass.append((truths[i][j][0][3] * reco_stdevs[0][3]) + reco_means[0][3])
            reco_lbar_mass.append((truths[i][j][1][3] * reco_stdevs[1][3]) + reco_means[1][3])
            reco_b_mass.append((truths[i][j][2][3] * reco_stdevs[2][3]) + reco_means[2][3])
            reco_bbar_mass.append((truths[i][j][3][3] * reco_stdevs[3][3]) + reco_means[3][3])
            reco_met_mass.append((truths[i][j][4][3] * reco_stdevs[4][3]) + reco_means[4][3])
            
    
    reconstructed_events_file = uproot.recreate(SAVE_DIR + 'bumblebee_reco.root')
    reconstructed_events_file["Bumblebee"] = {
            "gen_l_pt": gen_l_pt,
            "gen_lbar_pt": gen_lbar_pt,
            "gen_b_pt": gen_b_pt,
            "gen_bbar_pt": gen_bbar_pt,
            "gen_n_pt": gen_n_pt,
            "gen_nbar_pt": gen_nbar_pt,
            "gen_l_eta": gen_l_eta,
            "gen_lbar_eta": gen_lbar_eta,
            "gen_b_eta": gen_b_eta,
            "gen_bbar_eta": gen_bbar_eta,
            "gen_n_eta": gen_n_eta,
            "gen_nbar_eta": gen_nbar_eta,
            "gen_l_phi": gen_l_phi,
            "gen_lbar_phi": gen_lbar_phi,
            "gen_b_phi": gen_b_phi,
            "gen_bbar_phi": gen_bbar_phi,
            "gen_n_phi": gen_n_phi,
            "gen_nbar_phi": gen_nbar_phi,
            "gen_l_mass": gen_l_mass,
            "gen_lbar_mass": gen_lbar_mass,
            "gen_b_mass": gen_b_mass,
            "gen_bbar_mass": gen_bbar_mass,
            "gen_n_mass": gen_n_mass,
            "gen_nbar_mass": gen_nbar_mass,
            "pred_l_pt": pred_l_pt,
            "pred_lbar_pt": pred_lbar_pt,
            "pred_b_pt": pred_b_pt,
            "pred_bbar_pt": pred_bbar_pt,
            "pred_met_pt": pred_met_pt,
            "pred_l_eta": pred_l_eta,
            "pred_lbar_eta": pred_lbar_eta,
            "pred_b_eta": pred_b_eta,
            "pred_bbar_eta": pred_bbar_eta,
            "pred_met_eta": pred_met_eta,
            "pred_l_phi": pred_l_phi,
            "pred_lbar_phi": pred_lbar_phi,
            "pred_b_phi": pred_b_phi,
            "pred_bbar_phi": pred_bbar_phi,
            "pred_met_phi": pred_met_phi,
            "pred_l_mass": pred_l_mass,
            "pred_lbar_mass": pred_lbar_mass,
            "pred_b_mass": pred_b_mass,
            "pred_bbar_mass": pred_bbar_mass,
            "pred_met_mass": pred_met_mass,
            "reco_l_pt": reco_l_pt,
            "reco_lbar_pt": reco_lbar_pt,
            "reco_b_pt": reco_b_pt,
            "reco_bbar_pt": reco_bbar_pt,
            "reco_met_pt": reco_met_pt,
            "reco_l_eta": reco_l_eta,
            "reco_lbar_eta": reco_lbar_eta,
            "reco_b_eta": reco_b_eta,
            "reco_bbar_eta": reco_bbar_eta,
            "reco_met_eta": reco_met_eta,
            "reco_l_phi": reco_l_phi,
            "reco_lbar_phi": reco_lbar_phi,
            "reco_b_phi": reco_b_phi,
            "reco_bbar_phi": reco_bbar_phi,
            "reco_met_phi": reco_met_phi,
            "reco_l_mass": reco_l_mass,
            "reco_lbar_mass": reco_lbar_mass,
            "reco_b_mass": reco_b_mass,
            "reco_bbar_mass": reco_bbar_mass,
            "reco_met_mass": reco_met_mass,
        }
    
if post_as_pngs:
    # mass
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_l_mass, bins = N_BINS, label='gen l mass', histtype='step')
    _ = plt.hist(pred_l_mass, bins = bins, label='pred l mass', histtype='step')
    _ = plt.hist(reco_l_mass, bins = bins, label='reco l mass', histtype='step')
    plt.title('l mass')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lmasscomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_lbar_mass, bins = N_BINS, label='gen lbar mass', histtype='step')
    _ = plt.hist(pred_lbar_mass, bins = bins, label='pred lbar mass', histtype='step')
    _ = plt.hist(reco_lbar_mass, bins = bins, label='reco lbar mass', histtype='step')
    plt.title('lbar mass')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lbarmasscomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_b_mass, bins = N_BINS, label='gen b mass', histtype='step')
    _ = plt.hist(pred_b_mass, bins = bins, label='pred b mass', histtype='step')
    _ = plt.hist(reco_b_mass, bins = bins, label='reco b mass', histtype='step')
    plt.title('b mass')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bmasscomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_bbar_mass, bins = N_BINS, label='gen bbar mass', histtype='step')
    _ = plt.hist(pred_bbar_mass, bins = bins, label='pred bbar mass', histtype='step')
    _ = plt.hist(reco_bbar_mass, bins = bins, label='reco bbar mass', histtype='step')
    plt.title('bbar mass')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bbarmasscomp.png')
    plt.close()
    
    
    # eta
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_l_eta, bins = N_BINS, label='gen l eta', histtype='step')
    _ = plt.hist(pred_l_eta, bins = bins, label='pred l eta', histtype='step')
    _ = plt.hist(reco_l_eta, bins = bins, label='reco l eta', histtype='step')
    plt.title('l eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'letacomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_lbar_eta, bins = N_BINS, label='gen lbar eta', histtype='step')
    _ = plt.hist(pred_lbar_eta, bins = bins, label='pred lbar eta', histtype='step')
    _ = plt.hist(reco_lbar_eta, bins = bins, label='reco lbar eta', histtype='step')
    plt.title('lbar eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lbaretacomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_b_eta, bins = N_BINS, label='gen b eta', histtype='step')
    _ = plt.hist(pred_b_eta, bins = bins, label='pred b eta', histtype='step')
    _ = plt.hist(reco_b_eta, bins = bins, label='reco b eta', histtype='step')
    plt.title('b eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'betacomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_bbar_eta, bins = N_BINS, label='gen bbar eta', histtype='step')
    _ = plt.hist(pred_bbar_eta, bins = bins, label='pred bbar eta', histtype='step')
    _ = plt.hist(reco_bbar_eta, bins = bins, label='reco bbar eta', histtype='step')
    plt.title('bbar eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bbaretacomp.png')
    plt.close()
    
    
    # phi
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_l_phi, bins = N_BINS, label='gen l phi', histtype='step')
    _ = plt.hist(pred_l_phi, bins = bins, label='pred l phi', histtype='step')
    _ = plt.hist(reco_l_phi, bins = bins, label='reco l phi', histtype='step')
    plt.title('l phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_lbar_phi, bins = N_BINS, label='gen lbar phi', histtype='step')
    _ = plt.hist(pred_lbar_phi, bins = bins, label='pred lbar phi', histtype='step')
    _ = plt.hist(reco_lbar_phi, bins = bins, label='reco lbar phi', histtype='step')
    plt.title('lbar phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lbarphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_b_phi, bins = N_BINS, label='gen b phi', histtype='step')
    _ = plt.hist(pred_b_phi, bins = bins, label='pred b phi', histtype='step')
    _ = plt.hist(reco_b_phi, bins = bins, label='reco b phi', histtype='step')
    plt.title('b phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_bbar_phi, bins = N_BINS, label='gen bbar phi', histtype='step')
    _ = plt.hist(pred_bbar_phi, bins = bins, label='pred bbar phi', histtype='step')
    _ = plt.hist(reco_bbar_phi, bins = bins, label='reco bbar phi', histtype='step')
    plt.title('bbar phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bbarphicomp.png')
    plt.close()
    
    
    # pt
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_l_pt, bins = N_BINS, label='gen l pt', histtype='step')
    _ = plt.hist(pred_l_pt, bins = bins, label='pred l pt', histtype='step')
    _ = plt.hist(reco_l_pt, bins = bins, label='reco l pt', histtype='step')
    plt.title('l pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lptcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_lbar_pt, bins = N_BINS, label='gen lbar pt', histtype='step')
    _ = plt.hist(pred_lbar_pt, bins = bins, label='pred lbar pt', histtype='step')
    _ = plt.hist(reco_lbar_pt, bins = bins, label='reco lbar pt', histtype='step')
    plt.title('lbar pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lbarptcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_b_pt, bins = N_BINS, label='gen b pt', histtype='step')
    _ = plt.hist(pred_b_pt, bins = bins, label='pred b pt', histtype='step')
    _ = plt.hist(reco_b_pt, bins = bins, label='reco b pt', histtype='step')
    plt.title('b pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bptcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_bbar_pt, bins = N_BINS, label='gen bbar pt', histtype='step')
    _ = plt.hist(pred_bbar_pt, bins = bins, label='pred bbar pt', histtype='step')
    _ = plt.hist(reco_bbar_pt, bins = bins, label='reco bbar pt', histtype='step')
    plt.title('bbar pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bbarptcomp.png')
    plt.close()
    
    
    # met
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(reco_met_phi, bins = bins, label='reco met phi', histtype='step')
    _ = plt.hist(pred_met_phi, bins = bins, label='pred met phi', histtype='step')
    plt.title('met phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'metphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(pred_met_pt, bins = bins, label='pred met pt', histtype='step')
    _ = plt.hist(reco_met_pt, bins = bins, label='reco met pt', histtype='step')
    plt.title('met pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'metptcomp.png')
    plt.close()

    # wasserstein
    wd_l_pt = wasserstein_distance(reco_l_pt[:10000], pred_l_pt[:10000])
    wd_l_eta = wasserstein_distance(reco_l_eta[:10000], pred_l_eta[:10000])
    wd_l_phi = wasserstein_distance(reco_l_phi[:10000], pred_l_phi[:10000])
    wd_l_mass = wasserstein_distance(reco_l_mass[:10000], pred_l_mass[:10000])
    wd_lbar_pt = wasserstein_distance(reco_lbar_pt[:10000], pred_lbar_pt[:10000])
    wd_lbar_eta = wasserstein_distance(reco_lbar_eta[:10000], pred_lbar_eta[:10000])
    wd_lbar_phi = wasserstein_distance(reco_lbar_phi[:10000], pred_lbar_phi[:10000])
    wd_lbar_mass = wasserstein_distance(reco_lbar_mass[:10000], pred_lbar_mass[:10000])
    wd_b_pt = wasserstein_distance(reco_b_pt[:10000], pred_b_pt[:10000])
    wd_b_eta = wasserstein_distance(reco_b_eta[:10000], pred_b_eta[:10000])
    wd_b_phi = wasserstein_distance(reco_b_phi[:10000], pred_b_phi[:10000])
    wd_b_mass = wasserstein_distance(reco_b_mass[:10000], pred_b_mass[:10000])
    wd_bbar_pt = wasserstein_distance(reco_bbar_pt[:10000], pred_bbar_pt[:10000])
    wd_bbar_eta = wasserstein_distance(reco_bbar_eta[:10000], pred_bbar_eta[:10000])
    wd_bbar_phi = wasserstein_distance(reco_bbar_phi[:10000], pred_bbar_phi[:10000])
    wd_bbar_mass = wasserstein_distance(reco_bbar_mass[:10000], pred_bbar_mass[:10000])
    wd_met_pt = wasserstein_distance(reco_met_pt[:10000], pred_met_pt[:10000])
    wd_met_phi = wasserstein_distance(reco_met_phi[:10000], pred_met_phi[:10000])
    wd_labels = ["l_pt", "l_eta", "l_phi", "l_mass", "lbar_pt", "lbar_eta", "lbar_phi", "lbar_mass", 
                "b_pt", "b_eta", "b_phi", "b_mass", "bbar_pt", "bbar_eta", "bbar_phi", "bbar_mass", 
                "met_pt", "met_phi"]
    wd_scores = [wd_l_pt, wd_l_eta, wd_l_phi, wd_l_mass, wd_lbar_pt, wd_lbar_eta, wd_lbar_phi, wd_lbar_mass, 
                wd_b_pt, wd_b_eta, wd_b_phi, wd_b_mass, wd_bbar_pt, wd_bbar_eta, wd_bbar_phi, wd_bbar_mass, 
                wd_met_pt, wd_met_phi]
    fig = plt.figure(figsize=(10,8))
    plt.bar(wd_labels, wd_scores)
    plt.title("Relative component wasserstein scores")
    plt.xlabel("Particle component")
    plt.ylabel("Wasserstein distance between reco and pred_reco")
    plt.savefig(SAVE_DIR + 'wd_comparison.png')
    plt.close()
    
    

else:
    
    plt.rcParams["figure.figsize"] = [10, 8]
    plt.rcParams["figure.autolayout"] = True
    
    # mass
    fig1 = plt.figure()
    _, bins, _ = plt.hist(gen_l_mass, bins = N_BINS, label='gen l mass', histtype='step')
    _ = plt.hist(pred_l_mass, bins = bins, label='pred l mass', histtype='step')
    _ = plt.hist(reco_l_mass, bins = bins, label='reco l mass', histtype='step')
    plt.title('l mass')
    plt.yscale('log')
    plt.legend()
    
    fig2 = plt.figure()
    _, bins, _ = plt.hist(gen_lbar_mass, bins = N_BINS, label='gen lbar mass', histtype='step')
    _ = plt.hist(pred_lbar_mass, bins = bins, label='pred lbar mass', histtype='step')
    _ = plt.hist(reco_lbar_mass, bins = bins, label='reco lbar mass', histtype='step')
    plt.title('lbar mass')
    plt.yscale('log')
    plt.legend()
    
    fig3 = plt.figure()
    _, bins, _ = plt.hist(gen_b_mass, bins = N_BINS, label='gen b mass', histtype='step')
    _ = plt.hist(pred_b_mass, bins = bins, label='pred b mass', histtype='step')
    _ = plt.hist(reco_b_mass, bins = bins, label='reco b mass', histtype='step')
    plt.title('b mass')
    plt.yscale('log')
    plt.legend()
    
    fig4 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_mass, bins = N_BINS, label='gen bbar mass', histtype='step')
    _ = plt.hist(pred_bbar_mass, bins = bins, label='pred bbar mass', histtype='step')
    _ = plt.hist(reco_bbar_mass, bins = bins, label='reco bbar mass', histtype='step')
    plt.title('bbar mass')
    plt.yscale('log')
    plt.legend()
    
    fig5 = plt.figure()
    _, bins, _ = plt.hist(gen_t_mass, bins = N_BINS, label='gen t mass', histtype='step')
    _ = plt.hist(pred_t_mass, bins = bins, label='pred t mass', histtype='step')
    _ = plt.hist(reco_t_mass, bins = bins, label='reco t mass', histtype='step')
    plt.title('t mass')
    plt.yscale('log')
    plt.legend()
    
    fig6 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_mass, bins = N_BINS, label='gen tbar mass', histtype='step')
    _ = plt.hist(pred_tbar_mass, bins = bins, label='pred tbar mass', histtype='step')
    _ = plt.hist(reco_tbar_mass, bins = bins, label='reco tbar mass', histtype='step')
    plt.title('tbar mass')
    plt.yscale('log')
    plt.legend()
    
    # eta
    fig7 = plt.figure()
    _, bins, _ = plt.hist(gen_l_eta, bins = N_BINS, label='gen l eta', histtype='step')
    _ = plt.hist(pred_l_eta, bins = bins, label='pred l eta', histtype='step')
    _ = plt.hist(reco_l_eta, bins = bins, label='reco l eta', histtype='step')
    plt.title('l eta')
    plt.yscale('log')
    plt.legend()
    
    fig8 = plt.figure()
    _, bins, _ = plt.hist(gen_lbar_eta, bins = N_BINS, label='gen lbar eta', histtype='step')
    _ = plt.hist(pred_lbar_eta, bins = bins, label='pred lbar eta', histtype='step')
    _ = plt.hist(reco_lbar_eta, bins = bins, label='reco lbar eta', histtype='step')
    plt.title('lbar eta')
    plt.yscale('log')
    plt.legend()
    
    fig9 = plt.figure()
    _, bins, _ = plt.hist(gen_b_eta, bins = N_BINS, label='gen b eta', histtype='step')
    _ = plt.hist(pred_b_eta, bins = bins, label='pred b eta', histtype='step')
    _ = plt.hist(reco_b_eta, bins = bins, label='reco b eta', histtype='step')
    plt.title('b eta')
    plt.yscale('log')
    plt.legend()
    
    fig10 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_eta, bins = N_BINS, label='gen bbar eta', histtype='step')
    _ = plt.hist(pred_bbar_eta, bins = bins, label='pred bbar eta', histtype='step')
    _ = plt.hist(reco_bbar_eta, bins = bins, label='reco bbar eta', histtype='step')
    plt.title('bbar eta')
    plt.yscale('log')
    plt.legend()
    
    fig11 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_eta, bins = N_BINS, label='gen n eta', histtype='step')
    _ = plt.hist(pred_n_eta, bins = bins, label='pred n eta', histtype='step')
    plt.title('gen vs pred n eta')
    plt.yscale('log')
    plt.legend()
    
    fig12 = plt.figure()
    _, bins, _ = plt.hist(gen_nbar_eta, bins = N_BINS, label='gen nbar eta', histtype='step')
    _ = plt.hist(pred_nbar_eta, bins = bins, label='pred nbar eta', histtype='step')
    plt.title('gen vs pred nbar eta')
    plt.yscale('log')
    plt.legend()
    
    fig13 = plt.figure()
    _, bins, _ = plt.hist(gen_t_eta, bins = N_BINS, label='gen t eta', histtype='step')
    _ = plt.hist(pred_t_eta, bins = bins, label='pred t eta', histtype='step')
    _ = plt.hist(reco_t_eta, bins = bins, label='reco t eta', histtype='step')
    plt.title('t eta')
    plt.yscale('log')
    plt.legend()
    
    fig14 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_eta, bins = N_BINS, label='gen tbar eta', histtype='step')
    _ = plt.hist(pred_tbar_eta, bins = bins, label='pred tbar eta', histtype='step')
    _ = plt.hist(reco_tbar_eta, bins = bins, label='reco tbar eta', histtype='step')
    plt.title('tbar eta')
    plt.yscale('log')
    plt.legend()
    
    # phi
    fig15 = plt.figure()
    _, bins, _ = plt.hist(gen_l_phi, bins = N_BINS, label='gen l phi', histtype='step')
    _ = plt.hist(pred_l_phi, bins = bins, label='pred l phi', histtype='step')
    _ = plt.hist(reco_l_phi, bins = bins, label='reco l phi', histtype='step')
    plt.title('l phi')
    plt.yscale('log')
    plt.legend()
    
    fig16 = plt.figure()
    _, bins, _ = plt.hist(gen_lbar_phi, bins = N_BINS, label='gen lbar phi', histtype='step')
    _ = plt.hist(pred_lbar_phi, bins = bins, label='pred lbar phi', histtype='step')
    _ = plt.hist(reco_lbar_phi, bins = bins, label='reco lbar phi', histtype='step')
    plt.title('lbar phi')
    plt.yscale('log')
    plt.legend()
    
    fig17 = plt.figure()
    _, bins, _ = plt.hist(gen_b_phi, bins = N_BINS, label='gen b phi', histtype='step')
    _ = plt.hist(pred_b_phi, bins = bins, label='pred b phi', histtype='step')
    _ = plt.hist(reco_b_phi, bins = bins, label='reco b phi', histtype='step')
    plt.title('b phi')
    plt.yscale('log')
    plt.legend()
    
    fig18 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_phi, bins = N_BINS, label='gen bbar phi', histtype='step')
    _ = plt.hist(pred_bbar_phi, bins = bins, label='pred bbar phi', histtype='step')
    _ = plt.hist(reco_bbar_phi, bins = bins, label='reco bbar phi', histtype='step')
    plt.title('bbar phi')
    plt.yscale('log')
    plt.legend()
    
    fig19 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_phi, bins = N_BINS, label='gen n phi', histtype='step')
    _ = plt.hist(pred_n_phi, bins = bins, label='pred n phi', histtype='step')
    plt.title('gen vs pred n phi')
    plt.yscale('log')
    plt.legend()
    
    fig20 = plt.figure()
    _, bins, _ = plt.hist(gen_nbar_phi, bins = N_BINS, label='gen nbar phi', histtype='step')
    _ = plt.hist(pred_nbar_phi, bins = bins, label='pred nbar phi', histtype='step')
    plt.title('gen vs pred nbar phi')
    plt.yscale('log')
    plt.legend()
    
    fig21 = plt.figure()
    _, bins, _ = plt.hist(gen_t_phi, bins = N_BINS, label='gen t phi', histtype='step')
    _ = plt.hist(pred_t_phi, bins = bins, label='pred t phi', histtype='step')
    _ = plt.hist(reco_t_phi, bins = bins, label='reco t phi', histtype='step')
    plt.title('t phi')
    plt.yscale('log')
    plt.legend()
    
    fig22 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_phi, bins = N_BINS, label='gen tbar phi', histtype='step')
    _ = plt.hist(pred_tbar_phi, bins = bins, label='pred tbar phi', histtype='step')
    _ = plt.hist(reco_tbar_phi, bins = bins, label='reco tbar phi', histtype='step')
    plt.title('tbar phi')
    plt.yscale('log')
    plt.legend()
    
    # pt
    fig23 = plt.figure()
    _, bins, _ = plt.hist(gen_l_pt, bins = N_BINS, label='gen l pt', histtype='step')
    _ = plt.hist(pred_l_pt, bins = bins, label='pred l pt', histtype='step')
    _ = plt.hist(reco_l_pt, bins = bins, label='reco l pt', histtype='step')
    plt.title('l pt')
    plt.yscale('log')
    plt.legend()
    
    fig24 = plt.figure()
    _, bins, _ = plt.hist(gen_lbar_pt, bins = N_BINS, label='gen lbar pt', histtype='step')
    _ = plt.hist(pred_lbar_pt, bins = bins, label='pred lbar pt', histtype='step')
    _ = plt.hist(reco_lbar_pt, bins = bins, label='reco lbar pt', histtype='step')
    plt.title('lbar pt')
    plt.yscale('log')
    plt.legend()
    
    fig25 = plt.figure()
    _, bins, _ = plt.hist(gen_b_pt, bins = N_BINS, label='gen b pt', histtype='step')
    _ = plt.hist(pred_b_pt, bins = bins, label='pred b pt', histtype='step')
    _ = plt.hist(reco_b_pt, bins = bins, label='reco b pt', histtype='step')
    plt.title('b pt')
    plt.yscale('log')
    plt.legend()
    
    fig26 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_pt, bins = N_BINS, label='gen bbar pt', histtype='step')
    _ = plt.hist(pred_bbar_pt, bins = bins, label='pred bbar pt', histtype='step')
    _ = plt.hist(reco_bbar_pt, bins = bins, label='reco bbar pt', histtype='step')
    plt.title('bbar pt')
    plt.yscale('log')
    plt.legend()
    
    fig27 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_pt, bins = N_BINS, label='gen n pt', histtype='step')
    _ = plt.hist(pred_n_pt, bins = bins, label='pred n pt', histtype='step')
    plt.title('gen vs pred n pt')
    plt.yscale('log')
    plt.legend()
    
    fig28 = plt.figure()
    _, bins, _ = plt.hist(gen_nbar_pt, bins = N_BINS, label='gen nbar pt', histtype='step')
    _ = plt.hist(pred_nbar_pt, bins = bins, label='pred nbar pt', histtype='step')
    plt.title('gen vs pred nbar pt')
    plt.yscale('log')
    plt.legend()
    
    fig29 = plt.figure()
    _, bins, _ = plt.hist(gen_t_pt, bins = N_BINS, label='gen t pt', histtype='step')
    _ = plt.hist(pred_t_pt, bins = bins, label='pred t pt', histtype='step')
    _ = plt.hist(reco_t_pt, bins = bins, label='reco t pt', histtype='step')
    plt.title('t pt')
    plt.yscale('log')
    plt.legend()
    
    fig30 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_pt, bins = N_BINS, label='gen tbar pt', histtype='step')
    _ = plt.hist(pred_tbar_pt, bins = bins, label='pred tbar pt', histtype='step')
    _ = plt.hist(reco_tbar_pt, bins = bins, label='reco tbar pt', histtype='step')
    plt.title('tbar pt')
    plt.yscale('log')
    plt.legend()
    
    # px, py, pz
    fig31 = plt.figure()
    _, bins, _ = plt.hist(gen_t_px, bins = N_BINS, label='gen t px', histtype='step')
    _ = plt.hist(pred_t_px, bins = bins, label='pred t px', histtype='step')
    _ = plt.hist(reco_t_px, bins = bins, label='reco t px', histtype='step')
    plt.title('t px')
    plt.yscale('log')
    plt.legend()
    
    fig32 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_px, bins = N_BINS, label='gen tbar px', histtype='step')
    _ = plt.hist(pred_tbar_px, bins = bins, label='pred tbar px', histtype='step')
    _ = plt.hist(reco_tbar_px, bins = bins, label='reco tbar px', histtype='step')
    plt.title('tbar px')
    plt.yscale('log')
    plt.legend()
    
    fig33 = plt.figure()
    _, bins, _ = plt.hist(gen_t_py, bins = N_BINS, label='gen t py', histtype='step')
    _ = plt.hist(pred_t_py, bins = bins, label='pred t py', histtype='step')
    _ = plt.hist(reco_t_py, bins = bins, label='reco t py', histtype='step')
    plt.title('t py')
    plt.yscale('log')
    plt.legend()
    
    fig34 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_py, bins = N_BINS, label='gen tbar py', histtype='step')
    _ = plt.hist(pred_tbar_py, bins = bins, label='pred tbar py', histtype='step')
    _ = plt.hist(reco_tbar_py, bins = bins, label='reco tbar py', histtype='step')
    plt.title('tbar py')
    plt.yscale('log')
    plt.legend()
    
    fig35 = plt.figure()
    _, bins, _ = plt.hist(gen_t_pz, bins = N_BINS, label='gen t pz', histtype='step')
    _ = plt.hist(pred_t_pz, bins = bins, label='pred t pz', histtype='step')
    _ = plt.hist(reco_t_pz, bins = bins, label='reco t pz', histtype='step')
    plt.title('t pz')
    plt.yscale('log')
    plt.legend()
    
    fig36 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_pz, bins = N_BINS, label='gen tbar pz', histtype='step')
    _ = plt.hist(pred_tbar_pz, bins = bins, label='pred tbar pz', histtype='step')
    _ = plt.hist(reco_tbar_pz, bins = bins, label='reco tbar pz', histtype='step')
    plt.title('tbar pz')
    plt.yscale('log')
    plt.legend()
    
    
    # 2d plots
    fig37 = plt.figure()
    a = plt.hist2d(gen_t_px, pred_t_px, bins=100, range=[[-400, 400],[-400, 400]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top px plot')
    plt.xlabel(r'gen top px [GeV]')
    plt.ylabel(r'pred top px [GeV]')
    
    fig38 = plt.figure()
    a = plt.hist2d(gen_t_py, pred_t_py, bins=100, range=[[-400, 400],[-400, 400]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top py plot')
    plt.xlabel(r'gen top py [GeV]')
    plt.ylabel(r'pred top py [GeV]')
    
    fig39 = plt.figure()
    a = plt.hist2d(gen_t_pz, pred_t_pz, bins=100, range=[[-400, 400],[-400, 400]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top pz plot')
    plt.xlabel(r'gen top pz [GeV]')
    plt.ylabel(r'pred top pz [GeV]')
    
    
    fig40 = plt.figure()
    a = plt.hist2d(gen_t_mass, pred_t_mass, bins=100, range=[[50, 250],[0, 400]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top mass plot')
    plt.xlabel(r'gen top mass [GeV]')
    plt.ylabel(r'pred top mass [GeV]')
    
    fig41 = plt.figure()
    a = plt.hist2d(gen_t_phi, pred_t_phi, bins=100, range=[[-3.15, 3.15],[-3.15, 3.15]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top phi plot')
    plt.xlabel(r'gen top phi')
    plt.ylabel(r'pred top phi')
    
    fig42 = plt.figure()
    a = plt.hist2d(gen_t_eta, pred_t_eta, bins=100, range=[[-10, 10],[-10, 10]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top eta plot')
    plt.xlabel(r'gen top eta')
    plt.ylabel(r'pred top eta')
    
    fig43 = plt.figure()
    a = plt.hist2d(gen_t_pt, pred_t_pt, bins=100, range=[[0, 1600],[0, 1600]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top pT plot')
    plt.xlabel(r'gen top pT [GeV]')
    plt.ylabel(r'pred top pT [GeV]')
    
    # W stuff
    fig44 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_mass, bins = N_BINS, label='gen W+ mass', histtype='step')
    _ = plt.hist(pred_Wp_mass, bins = bins, label='pred W+ mass', histtype='step')
    plt.title('gen vs pred W+ mass')
    plt.yscale('log')
    plt.legend()
    
    fig45 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_mass, bins = N_BINS, label='gen W- mass', histtype='step')
    _ = plt.hist(pred_Wm_mass, bins = bins, label='pred W- mass', histtype='step')
    plt.title('gen vs pred W- mass')
    plt.yscale('log')
    plt.legend()
    
    fig46 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_eta, bins = N_BINS, label='gen W+ eta', histtype='step')
    _ = plt.hist(pred_Wp_eta, bins = bins, label='pred W+ eta', histtype='step')
    plt.title('gen vs pred W+ eta')
    plt.yscale('log')
    plt.legend()
    
    fig47 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_eta, bins = N_BINS, label='gen W- eta', histtype='step')
    _ = plt.hist(pred_Wm_eta, bins = bins, label='pred W- eta', histtype='step')
    plt.title('gen vs pred W- eta')
    plt.yscale('log')
    plt.legend()
    
    fig48 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_phi, bins = N_BINS, label='gen W+ phi', histtype='step')
    _ = plt.hist(pred_Wp_phi, bins = bins, label='pred W+ phi', histtype='step')
    plt.title('gen vs pred W+ phi')
    plt.yscale('log')
    plt.legend()
    
    fig49 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_phi, bins = N_BINS, label='gen W- phi', histtype='step')
    _ = plt.hist(pred_Wm_phi, bins = bins, label='pred W- phi', histtype='step')
    plt.title('gen vs pred W- phi')
    plt.yscale('log')
    plt.legend()
    
    fig50 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_pt, bins = N_BINS, label='gen W+ pt', histtype='step')
    _ = plt.hist(pred_Wp_pt, bins = bins, label='pred W+ pt', histtype='step')
    plt.title('gen vs pred W+ pt')
    plt.yscale('log')
    plt.legend()
    
    fig51 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_pt, bins = N_BINS, label='gen W- pt', histtype='step')
    _ = plt.hist(pred_Wm_pt, bins = bins, label='pred W- pt', histtype='step')
    plt.title('gen vs pred W- pt')
    plt.yscale('log')
    plt.legend()
    
    fig52 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_px, bins = N_BINS, label='gen W+ px', histtype='step')
    _ = plt.hist(pred_Wp_px, bins = bins, label='pred W+ px', histtype='step')
    plt.title('gen vs pred W+ px')
    plt.yscale('log')
    plt.legend()
    
    fig53 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_px, bins = N_BINS, label='gen W- px', histtype='step')
    _ = plt.hist(pred_Wm_px, bins = bins, label='pred W- px', histtype='step')
    plt.title('gen vs pred W- px')
    plt.yscale('log')
    plt.legend()
    
    fig54 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_py, bins = N_BINS, label='gen W+ py', histtype='step')
    _ = plt.hist(pred_Wp_py, bins = bins, label='pred W+ py', histtype='step')
    plt.title('gen vs pred W+ py')
    plt.yscale('log')
    plt.legend()
    
    fig55 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_py, bins = N_BINS, label='gen W- py', histtype='step')
    _ = plt.hist(pred_Wm_py, bins = bins, label='pred W- py', histtype='step')
    plt.title('gen vs pred W- py')
    plt.yscale('log')
    plt.legend()
    
    fig56 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_pz, bins = N_BINS, label='gen W+ pz', histtype='step')
    _ = plt.hist(pred_Wp_pz, bins = bins, label='pred W+ pz', histtype='step')
    plt.title('gen vs pred W+ pz')
    plt.yscale('log')
    plt.legend()
    
    fig57 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_pz, bins = N_BINS, label='gen W- pz', histtype='step')
    _ = plt.hist(pred_Wm_pz, bins = bins, label='pred W- pz', histtype='step')
    plt.title('gen vs pred W- pz')
    plt.yscale('log')
    plt.legend()
    
    # met
    fig58 = plt.figure()
    _, bins, _ = plt.hist(gen_met_phi, bins = N_BINS, label='gen met phi', histtype='step')
    _ = plt.hist(pred_met_phi, bins = bins, label='pred met phi', histtype='step')
    _ = plt.hist(reco_met_phi, bins = bins, label='reco met phi', histtype='step')
    plt.title('met phi')
    plt.yscale('log')
    plt.legend()
    
    fig59 = plt.figure()
    _, bins, _ = plt.hist(gen_met_pt, bins = N_BINS, label='gen met pt', histtype='step')
    _ = plt.hist(pred_met_pt, bins = bins, label='pred met pt', histtype='step')
    _ = plt.hist(reco_met_pt, bins = bins, label='reco met pt', histtype='step')
    plt.title('met pt')
    plt.yscale('log')
    plt.legend()
    
    # ttbar
    fig60 = plt.figure()
    _, bins, _ = plt.hist(gen_ttbar_pt, bins = N_BINS, label='gen ttbar pt', histtype='step')
    _ = plt.hist(pred_ttbar_pt, bins = bins, label='pred ttbar pt', histtype='step')
    _ = plt.hist(reco_ttbar_pt, bins = bins, label='reco ttbar pt', histtype='step')
    plt.title('ttbar pt')
    plt.yscale('log')
    plt.legend()
    
    fig61 = plt.figure()
    _, bins, _ = plt.hist(gen_ttbar_eta, bins = N_BINS, label='gen ttbar eta', histtype='step')
    _ = plt.hist(pred_ttbar_eta, bins = bins, label='pred ttbar eta', histtype='step')
    _ = plt.hist(reco_ttbar_eta, bins = bins, label='reco ttbar eta', histtype='step')
    plt.title('ttbar eta')
    plt.yscale('log')
    plt.legend()
    
    fig62 = plt.figure()
    _, bins, _ = plt.hist(gen_ttbar_phi, bins = N_BINS, label='gen ttbar phi', histtype='step')
    _ = plt.hist(pred_ttbar_phi, bins = bins, label='pred ttbar phi', histtype='step')
    _ = plt.hist(reco_ttbar_phi, bins = bins, label='reco ttbar phi', histtype='step')
    plt.title('ttbar phi')
    plt.yscale('log')
    plt.legend()
    
    fig63 = plt.figure()
    _, bins, _ = plt.hist(gen_ttbar_mass, bins = N_BINS, label='gen ttbar mass', histtype='step')
    _ = plt.hist(pred_ttbar_mass, bins = bins, label='pred ttbar mass', histtype='step')
    _ = plt.hist(reco_ttbar_mass, bins = bins, label='reco ttbar mass', histtype='step')
    plt.title('ttbar mass')
    plt.yscale('log')
    plt.legend()
    
    fig64 = plt.figure()
    a = plt.hist2d(gen_ttbar_mass, pred_ttbar_mass, bins=100, range=[[150, 800],[150, 800]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D ttbar mass plot')
    plt.xlabel(r'gen ttbar mass [GeV]')
    plt.ylabel(r'pred ttbar mass [GeV]')
    
    fig65 = plt.figure()
    a = plt.hist2d(gen_ttbar_phi, pred_ttbar_phi, bins=100, range=[[-3.15, 3.15],[-3.15, 3.15]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D ttbar phi plot')
    plt.xlabel(r'gen ttbar phi')
    plt.ylabel(r'pred ttbar phi')
    
    fig66 = plt.figure()
    a = plt.hist2d(gen_ttbar_eta, pred_ttbar_eta, bins=100, range=[[-10, 10],[-10, 10]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D ttbar eta plot')
    plt.xlabel(r'gen ttbar eta')
    plt.ylabel(r'pred ttbar eta')
    
    fig67 = plt.figure()
    a = plt.hist2d(gen_ttbar_pt, pred_ttbar_pt, bins=100, range=[[0, 1000],[0, 1000]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D ttbar pT plot')
    plt.xlabel(r'gen ttbar pT [GeV]')
    plt.ylabel(r'pred ttbar pT [GeV]')
    
    
    # make pdf
    
    p = PdfPages(SAVE_DIR + 'multi_plot.pdf')
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    
    for fig in figs:
        fig.savefig(p, format = 'pdf')
    p.close()
    
    

