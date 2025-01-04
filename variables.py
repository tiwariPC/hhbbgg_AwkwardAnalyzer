import copy

regions = ["preselection", "selection", "srbbgg", "srbbggMET", "crantibbgg", "crbbantigg"]
vardict = {
    "dibjet_mass": "dibjet_mass",
    "diphoton_mass": "diphoton_mass",
    "bbgg_mass": "bbgg_mass",
    "dibjet_pt": "dibjet_pt",
    "diphoton_pt": "diphoton_pt",
    "bbgg_pt": "bbgg_pt",
    "bbgg_eta": "bbgg_eta",
    "bbgg_phi": "bbgg_phi",
    "lead_pho_pt": "lead_pho_pt",
    "sublead_pho_pt": "sublead_pho_pt",
    "lead_pho_eta": "lead_pho_eta",
    "lead_pho_phi": "lead_pho_phi",
    "sublead_pho_eta": "sublead_pho_eta",
    "sublead_pho_phi": "sublead_pho_phi",
    "diphoton_eta": "diphoton_eta",
    "diphoton_phi": "diphoton_phi",
    "dibjet_eta": "dibjet_eta",
    "dibjet_phi": "dibjet_phi",
    # ----bjet-----
    "lead_bjet_pt": "lead_bjet_pt",
    "sublead_bjet_pt": "sublead_bjet_pt",
    "lead_bjet_eta": "lead_bjet_eta",
    "lead_bjet_phi": "lead_bjet_phi",
    "sublead_bjet_eta": "sublead_bjet_eta",
    "sublead_bjet_phi": "sublead_bjet_phi",
    "sublead_bjet_PNetB": "sublead_bjet_PNetB",
    "lead_bjet_PNetB": "lead_bjet_PNetB",
    "CosThetaStar_gg": "CosThetaStar_gg",
    "CosThetaStar_jj": "CosThetaStar_jj",
    "CosThetaStar_CS": "CosThetaStar_CS",
    "DeltaR_jg_min": "DeltaR_jg_min",
    "pholead_PtOverM": "pholead_PtOverM",
    "phosublead_PtOverM": "phosublead_PtOverM",
    "FirstJet_PtOverM": "FirstJet_PtOverM",
    "SecondJet_PtOverM": "SecondJet_PtOverM",
    "lead_pt_over_diphoton_mass": "lead_pt_over_diphoton_mass",
    "sublead_pt_over_diphoton_mass": "sublead_pt_over_diphoton_mass",
    "lead_pt_over_dibjet_mass": "lead_pt_over_dibjet_mass",
    "sublead_pt_over_dibjet_mass": "sublead_pt_over_dibjet_mass",
    "diphoton_bbgg_mass": "diphoton_bbgg_mass",
    "dibjet_bbgg_mass": "dibjet_bbgg_mass",
    "puppiMET_pt": "puppiMET_pt",
    "puppiMET_phi": "puppiMET_phi",
    "puppiMET_phiJERDown": "puppiMET_phiJERDown",
    "puppiMET_phiJERUp": "puppiMET_phiJERUp",
    "puppiMET_phiJESDown": "puppiMET_phiJESDown",
    "puppiMET_phiJESUp": "puppiMET_phiJESUp",
    "puppiMET_phiUnclusteredDown": "puppiMET_phiUnclusteredDown",
    "puppiMET_phiUnclusteredUp": "puppiMET_phiUnclusteredUp",
    "puppiMET_ptJERDown":"puppiMET_ptJERDown",
    "puppiMET_ptJERUp":"puppiMET_ptJERUp",
    "puppiMET_ptJESDown":"puppiMET_ptJESDown",
    "puppiMET_ptJESUp":"puppiMET_ptJESUp",
    "lead_pho_mvaID_WP90":"lead_pho_mvaID_WP90",
    "lead_pho_mvaID_WP80":"lead_pho_mvaID_WP80",
    "sublead_pho_mvaID_WP90":"sublead_pho_mvaID_WP90",
    "sublead_pho_mvaID_WP80":"sublead_pho_mvaID_WP80",
    "lead_pho_mvaID":"lead_pho_mvaID",
    "sublead_pho_mvaID":"sublead_pho_mvaID",
    
}

variables_common = {
    "preselection": [
        "dibjet_mass",
        "diphoton_mass",
        "bbgg_mass",
        "dibjet_pt",
        "diphoton_pt",
        "bbgg_pt",
        "lead_pho_pt",
        "sublead_pho_pt",
        "bbgg_eta",
        "bbgg_phi",
        "lead_pho_eta",
        "lead_pho_phi",
        "sublead_pho_eta",
        "sublead_pho_phi",
        "diphoton_eta",
        "diphoton_phi",
        "dibjet_eta",
        "dibjet_phi",
        "lead_bjet_pt",
        "sublead_bjet_pt",
        "lead_bjet_eta",
        "lead_bjet_phi",
        "sublead_bjet_eta",
        "sublead_bjet_phi",
        "sublead_bjet_PNetB",
        "lead_bjet_PNetB",
        "CosThetaStar_gg",
        "CosThetaStar_jj",
        "CosThetaStar_CS",
        "DeltaR_jg_min",
        "pholead_PtOverM",
        "phosublead_PtOverM",
        "FirstJet_PtOverM",
        "SecondJet_PtOverM",
        "lead_pt_over_diphoton_mass",
        "sublead_pt_over_diphoton_mass",
        "lead_pt_over_dibjet_mass",
        "sublead_pt_over_dibjet_mass",
        "diphoton_bbgg_mass",
        "dibjet_bbgg_mass",
        "lead_pho_mvaID_WP90",
        "lead_pho_mvaID_WP80",
        "sublead_pho_mvaID_WP90",
        "sublead_pho_mvaID_WP80",
        "lead_pho_mvaID",
        "sublead_pho_mvaID",
        "puppiMET_pt", 
        "puppiMET_phi", 
        "puppiMET_phiJERDown",
        "puppiMET_phiJERUp",
        "puppiMET_phiJESDown", 
        "puppiMET_phiJESUp", 
        "puppiMET_phiUnclusteredDown", 
        "puppiMET_phiUnclusteredUp",
        "puppiMET_ptJERDown", 
        "puppiMET_ptJERUp", 
        "puppiMET_ptJESDown", 
        "puppiMET_ptJESUp"
    ]
}

for ireg in regions:
    print(ireg)
    variables_common[ireg] = copy.deepcopy(variables_common["preselection"])

srbbggMET = ["puppiMET_pt", "puppiMET_phi", "puppiMET_phiJERDown", "puppiMET_phiJERUp","puppiMET_phiJESDown", "puppiMET_phiJESUp", "puppiMET_phiUnclusteredDown", "puppiMET_phiUnclusteredUp", "puppiMET_ptJERDown", "puppiMET_ptJERUp", "puppiMET_ptJESDown", "puppiMET_ptJESUp"]
variables_common["srbbggMET"] = variables_common["srbbggMET"] + srbbggMET
