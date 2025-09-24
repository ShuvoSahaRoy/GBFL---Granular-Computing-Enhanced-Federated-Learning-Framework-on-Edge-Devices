SEED = 42

num_clients = 10
participants = 3
CR = 50

local_epoch = 5
non_iid = 0
mu = 0.01  # Proximal term coefficient for FedProx
alpha = 0.5
convergence_check = False

min_samples_per_client= 20
fallback_delbals = [7, 4, 2]
pur=0.99


dataset_path = ".\\data\\"
one_zero_dataset = ['phishing.csv',"qsar_oral_toxicity.csv", '2dplanes.csv',  'A9a.csv', 'adult.csv']
continuous_datasets = ['magic_gamma_telescope.csv','fried.csv', 'Run_or_walk_information.csv','skin_nonskin.csv','hepmass.csv', 'higgs.csv','susy.csv']

# all_datasets = one_zero_dataset + continuous_datasets
# all_datasets = ['A9a.csv','adult.csv','2dplanes.csv','fried.csv','hepmass.csv','magic_gamma_telescope.csv','phishing.csv','qsar_oral_toxicity.csv']
# all_datasets = ['A9a.csv','adult.csv','2dplanes.csv','hepmass.csv','phishing.csv','qsar_oral_toxicity.csv']
all_datasets =  ['phishing.csv']

# ['fedavg_lr','fedavg_svm', 'fedavg_svm_gb', 'fedavg_lr_gb','fedavg_unisvm', 'fedavg_unisvm_gb']
# aggregations = ['fedavg_lr','fedavg_lr_gb', 'fedavg_unisvm', 'fedavg_unisvm_gb']
# aggregations = ['fedavg_lr','fedavg_lr_gb','fedprox_lr','fedprox_lr_gb',]
aggregations = ['fedavg_lr','fedavg_lr_gb','fedavg_nn','fedavg_nn_gb']



if 'fedavg_lr_gb' or 'fedprox_lr_gb' or 'fedavg_NN_gb' or 'fedprox_NN_gb' in aggregations:
    make_gb = True
else:   
    make_gb = False

save = 0
save_log = 0