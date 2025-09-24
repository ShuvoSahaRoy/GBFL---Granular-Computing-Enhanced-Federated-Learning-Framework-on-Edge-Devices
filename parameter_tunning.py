from config import *                       
from Base.utils.utils import setup_logger
from Base.utils.utils import set_seed
from Base.utils.load_data import load_datasets
from Base.FedAvg.fedavg import main_fedavg
from Base.utils.plot_result import plot_lr_tuning
from Base.UniCSL.unicsl import main_unicsl
from Base.FedAvg.fedavgNN import main_fedavgNN
import copy

set_seed()
log_file, tee = setup_logger(save_log=save_log,log_dir="results/logs/parameter_tunning")
lr_list = [0.9,0.5,0.25,0.1,0.05,0.01,0.001,0.0001]

results_list = []

for dataset in all_datasets:
    train_data_list, test_data, gb_train_list, gb_making_time = load_datasets(dataset)

    for aggregation in aggregations:
        accuracies = []
        for lr in lr_list:
            print(f"""Processing {dataset.split('.')[0]} dataset with {num_clients} clients {participants} participants non_iid={non_iid} 
                   alpha={alpha} CR={CR} local_epoch={local_epoch} lr {lr}""")
            # fedavg with svm and original data point
            # if aggregation == 'fedavg_lr':
            #     FedAvg_LR = main_fedavg(copy.deepcopy(train_data_list), test_data.copy(), lr, dataset, base_model='LR',gb=False)
            #     print(f"FedAvg with logistic regression and original data points Simulation complete on {dataset}")
            #     accuracies.append(FedAvg_LR)

            # elif aggregation == 'fedavg_lr_gb':
            #     FedAvg_LR_GB = main_fedavg(copy.deepcopy(gb_train_list), test_data.copy(), lr, dataset, base_model='LR', gb=True)
            #     print(f"FedAvg with logistic regression and GB Simulation complete on {dataset}")
            #     accuracies.append(FedAvg_LR_GB)

            # elif aggregation == 'fedavg_svm':
            #     FedAvg_SVM = main_fedavg(copy.deepcopy(train_data_list), test_data.copy(), lr, dataset, base_model='SVM')
            #     print(f"FedAvg with SVM and original data points Simulation complete on {dataset}")
            #     accuracies.append(FedAvg_SVM)

            # elif aggregation == 'fedavg_unisvm':
            #     FedAvg_UniSVM = main_unicsl(copy.deepcopy(train_data_list), test_data.copy(), lr, dataset, gb=False)
            #     print(f"FedAvg with UniSVM and original data points Simulation complete on {dataset}")
            #     accuracies.append(FedAvg_UniSVM)

            # elif aggregation == 'fedavg_unisvm_gb':
            #     FedAvg_UniSVM_GB = main_unicsl(copy.deepcopy(gb_train_list), test_data.copy(), lr, dataset, gb=True)
            #     print(f"FedAvg with UniSVM and GB Simulation complete on {dataset}")
            #     accuracies.append(FedAvg_UniSVM_GB)

            if aggregation == 'fedavg_NN':
                FedAvg_NN = main_fedavgNN(copy.deepcopy(train_data_list), test_data.copy(), lr, dataset, base_model='NN',gb=False)
                print(f"FedAvg with nn and original data points Simulation complete on {dataset}")
                accuracies.append(FedAvg_NN)

            elif aggregation == 'fedavg_NN_gb':
                FedAvg_NN = main_fedavgNN(copy.deepcopy(gb_train_list), test_data.copy(), lr, dataset, base_model='NN',gb=True)
                print(f"FedAvg with nn and GB data points Simulation complete on {dataset}")
                accuracies.append(FedAvg_NN)

        plot_lr_tuning(accuracies, lr_list, dataset, aggregation)

if log_file:
    log_file.close()
    tee.close()