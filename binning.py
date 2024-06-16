import copy
binning={}
binning["preselection"]={
    "dibjet_mass":[25,80,180],
    "diphoton_mass":[25,80,180],
    "bbgg_mass":[25,150,500],
    "dibjet_pt":[25,30,180],
    "diphoton_pt":[25,30,180],
    "bbgg_pt":[25,30,500],
    "bbgg_eta":[10,-3,3],
    "bbgg_phi":[10,-3.14,3.14],
    "lead_pho_pt":[25,35,200],
    "lead_pho_eta":[10,-3,3],
    "lead_pho_phi":[10,-3.14,3.14],
    "sublead_pho_pt":[25,25,200],
    "sublead_pho_eta":[10,-3,3],
    "sublead_pho_phi":[10,-3.14,3.14],
    "dibjet_eta":[10,-3,3],
    "dibjet_phi":[10,-3.14,3.14],
    "diphoton_eta":[10,-3,3],
    "diphoton_phi":[10,-3.14,3.14],
}

binning["selection"] =copy.deepcopy(binning["preselection"])
