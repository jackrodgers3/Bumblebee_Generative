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

# *** very important parameters *** edit at your own risk
N_BINS = 40
BASE_DIR = r'/depot/cms/top/jprodger/Bumblebee/src/Experiment121324/output/'
SAVE_DIR = BASE_DIR

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

'''
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
'''
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
fig = plt.figure(figsize=(16,8))
plt.bar(wd_labels, wd_scores)
plt.title("Relative component wasserstein scores")
plt.xlabel("Particle component")
plt.ylabel("Wasserstein distance between reco and pred_reco")
plt.savefig(SAVE_DIR + 'wd_comparison.png')
plt.close()
