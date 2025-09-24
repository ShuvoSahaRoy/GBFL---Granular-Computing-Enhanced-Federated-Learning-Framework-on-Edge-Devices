import pandas as pd
from config import *                       
from Base.utils.utils import setup_logger
from Base.utils.utils import set_seed
from Base.utils.load_data import load_datasets
from Base.FedAvg.fedavg import main_fedavg
from Base.utils.plot_result import plot_all
from Base.FedProx.fedprox import main_fedprox
from Base.FedAvg.fedavgNN import main_fedavgNN
import copy, os
import time


set_seed()
log_file, tee = setup_logger(save_log=save_log)
lr_df = pd.read_excel("lr.xlsx")
results_list = []

for dataset in all_datasets:
    train_data_list, test_data, gb_train_list, gb_making_time = load_datasets(dataset)
    
    for aggregation in aggregations:
        lr = lr_df.loc[lr_df['dataset'] == dataset, aggregation].iloc[0]
        # lr = 0.1

        start_time = time.time()

        # fedavg with svm and original data point
        if aggregation == 'fedavg_lr':
            FedAvg_LR = main_fedavg(copy.deepcopy(train_data_list), test_data.copy(), lr, dataset, base_model='LR',gb=False)
            print(f"FedAvg with logistic regression and original data points Simulation complete on {dataset}")

        elif aggregation == 'fedavg_lr_gb':
            FedAvg_LR_GB = main_fedavg(copy.deepcopy(gb_train_list), test_data.copy(), lr, dataset, base_model='LR', gb=True)
            print(f"FedAvg with logistic regression and GB Simulation complete on {dataset}")

        # elif aggregation == 'fedprox_lr':
        #     FedProx_LR = main_fedprox(copy.deepcopy(train_data_list), test_data.copy(), lr, dataset, base_model='LR',gb=False)
        #     print(f"FedAvg with logistic regression and original data points Simulation complete on {dataset}")

        # elif aggregation == 'fedprox_lr_gb':
        #     FedProx_LR_GB = main_fedprox(copy.deepcopy(gb_train_list), test_data.copy(), lr, dataset, base_model='LR', gb=True)
        #     print(f"FedAvg with UniSVM and GB Simulation complete on {dataset}")
        
        elif aggregation == 'fedavg_nn':
            FedAvg_NN = main_fedavgNN(copy.deepcopy(train_data_list), test_data.copy(), lr, dataset, base_model='NN',gb=False)
            print(f"FedAvg with NN and original data points Simulation complete on {dataset}")

        elif aggregation == 'fedavg_nn_gb':
            FedAvg_NN_GB = main_fedavgNN(copy.deepcopy(gb_train_list), test_data.copy(), lr, dataset, base_model='NN',gb=True)
            print(f"FedAvg with NN and GB data points Simulation complete on {dataset}")


        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time for {aggregation} on {dataset}: {elapsed_time:.2f} seconds")
        # Store the result
        results_list.append({"Dataset": dataset,"Algorithm": aggregation,"Non_IID": non_iid,"Execution Time (s)": elapsed_time,"GB processing": gb_making_time})

    # print(f"Gb making time: {gb_making_time:.2f} seconds")
    results_df = pd.DataFrame(results_list)
    # print(results_df)

    global_acc = [FedAvg_LR, FedAvg_LR_GB, FedAvg_NN, FedAvg_NN_GB]
    algo_list = ["M1", "GB_M1", "M2", "GB_M2"]

    # accuracy_lists, algo_names, dataset y_label='Not provided'
    plot_all(global_acc, algo_list, dataset)

# Save results to Excel
if save:
    # Build DataFrame
    alg_names = ["M1", "GB_M1", "M2", "GB_M2"]
    acc_dict = {"Round": list(range(1, len(global_acc[0]) + 1))}
    for name, acc in zip(alg_names, global_acc):
        acc_dict[name] = acc

    acc_df = pd.DataFrame(acc_dict)

    # Save combined into one sheet (append if file exists)
    output_file = "results/global_accuracies.xlsx"
    if os.path.exists(output_file):
        existing = pd.read_excel(output_file)
        combined = pd.concat([existing, acc_df], ignore_index=True)
        combined.to_excel(output_file, index=False)
    else:
        acc_df.to_excel(output_file, index=False)

    print(f"Saved global accuracies for {dataset} to '{output_file}'")

if save:
    output_file = "results/execution_times.xlsx"
    results_df = pd.DataFrame(results_list)
    if os.path.exists(output_file):
        # Load existing Excel file and append new data
        existing_df = pd.read_excel(output_file)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    results_df.to_excel(output_file, index=False)
    print(f"Execution times appended to '{output_file}'.")

if log_file:
    log_file.close()
    tee.close()