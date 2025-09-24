import numpy as np
from config import *
from Base.classifier import logistic_regression
from Base.classifier import svm
from Base.utils.utils import process_gb_data_lr, check_convergence


def SERVER(g_model, local_updates, lr, n):
    w = []
    for update in local_updates:
        w.append(update[0] * (update[1]/n))
    
    w = np.array(w)
    
    global_model = g_model - lr * np.sum(w,axis=0)
    # global_model = np.sum(w,axis=0)

    # global_model = global_model/np.linalg.norm(global_model) 
    return global_model


def client_update(model, data, lr, base_model):
    global_model = model.copy()
    if base_model == 'LR':
        model = logistic_regression.train(model, data, lr)
    elif base_model == 'SVM':
        model = svm.train(model, data, lr)
    
    del_w = global_model - model
    # del_w = model
    local_update = (del_w, data.shape[0])
    return local_update

def main_fedavg(train_data_list, test_data, lr, dataset, base_model, gb=False):

    if gb:
        train_data_list = process_gb_data_lr(train_data_list)

    features = train_data_list[0].shape[1] - 1

    # Initialize parameters
    global_parameters = np.random.rand(features + 1)

    global_accuracy = []

    for round in range(CR):

        # select random clients
        client_set = np.random.choice(num_clients, size=int(participants), replace=False)

        n = 0
        local_updates = []

        for client in client_set:
            client_data = train_data_list[client]

            local_update = client_update(global_parameters.copy(), client_data, lr, base_model)

            local_updates.append(local_update)
            n = n + len(client_data[1])


        global_parameters = SERVER(global_parameters,local_updates, lr, n)
        
        if base_model == 'LR':
            global_acc = logistic_regression.test(global_parameters,test_data)
        elif base_model == 'SVM':
            global_acc = svm.test(global_parameters,test_data)

        global_accuracy.append(global_acc)

        print(f" {'-' * 20} communication round {round+1} {'-' * 20}")
        print(f"{20 * '-'}{dataset} Global accuracy {global_acc:.4f}{20 * '-'}\n")

        # if convergence_check:
        #     if check_convergence(global_accuracy, window_size=20, tolerance=0.05 if lr > 0.1 else 0.01):
        #         lr *= 0.5  # Reduce learning rate
            # if check_convergence(global_accuracy, window_size=20, tolerance=1):
            #     print("Converged")
            #     # break

    return global_accuracy