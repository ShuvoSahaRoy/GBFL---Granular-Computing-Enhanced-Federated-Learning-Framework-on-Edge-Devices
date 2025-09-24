import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from .plot_result import plot_data_distribution
from .gen_ball import gen_balls
import time
from config import continuous_datasets, SEED, num_clients, non_iid, alpha, make_gb, fallback_delbals, pur, dataset_path,  min_samples_per_client

def convert_to_granular_balls(train_data):
    
    # fallback_delbals = [2]

    balls = None
    for delbals in fallback_delbals:
        try:
            balls = gen_balls(train_data, pur=pur, delbals=delbals)
            break  # Success, no need to try further
        except Exception as e:
            print(f"Error with delbals={delbals}. Error: {e}")


    centers = np.array([ball[0] for ball in balls])
    # print(centers.shape)
    radii = np.array([ball[1] for ball in balls])
    # print(radii.shape)
    labels = np.array([ball[2] for ball in balls]) 
    # print(labels.shape)

    gb_data = pd.DataFrame(centers)
    gb_data['radius'] = radii
    gb_data['label'] = labels

    return gb_data


# def noniid_distribution(data):

    prng = np.random.default_rng(seed=SEED)
    total_samples = len(data)
    indices = data.index.to_numpy()
    prng.shuffle(indices)  # Shuffle indices to randomize the order
    
    # Step 1: Allocate minimum samples to each client
    min_samples_assigned = min_samples_per_client * num_clients
    if min_samples_assigned > total_samples:
        raise ValueError("Not enough samples to assign minimum samples to each client.")
    
    # Assign minimum samples to each client
    client_splits = []
    remaining_samples = total_samples - min_samples_assigned
    for i in range(num_clients):
        client_splits.append(indices[i * min_samples_per_client: (i + 1) * min_samples_per_client])

    remaining_indices = indices[min_samples_assigned:]
    
    # Step 2: Apply Dirichlet distribution to remaining samples
    if remaining_samples > 0:
        proportions = prng.dirichlet([alpha] * num_clients)
        proportions = proportions / proportions.sum()  # Normalize to sum to 1
        
        split_sizes = (proportions * remaining_samples).astype(int)
        # Adjust split sizes to ensure all remaining samples are assigned
        split_sizes[-1] += remaining_samples - split_sizes.sum()

        # Split the remaining indices among clients according to the Dirichlet distribution
        split_points = np.cumsum(split_sizes)[:-1]
        remaining_client_splits = np.split(remaining_indices, split_points)
        
        # Add the remaining samples to the clients
        for i in range(num_clients):
            client_splits[i] = np.concatenate([client_splits[i], remaining_client_splits[i]])

    # Ensure no data duplication by resetting the indices
    client_dataframes = [data.loc[split].reset_index(drop=True) for split in client_splits]

    # Verify no samples are lost
    total_samples = sum(len(client_data) for client_data in client_dataframes)
    assert total_samples == len(data), f"Sample mismatch: {total_samples} != {len(data)}"

    return client_dataframes


def noniid_distribution(data):
    # prng = np.random.default_rng(seed=SEED)
    total_samples = len(data)
    labels = data.iloc[:, -1]
    class_0_indices = data[labels == 0].index.to_numpy()
    class_1_indices = data[labels == 1].index.to_numpy()
    # prng.shuffle(class_0_indices)
    # prng.shuffle(class_1_indices)
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)


    min_samples_assigned = min_samples_per_client * num_clients
    if min_samples_assigned > total_samples:
        raise ValueError("Not enough samples for minimum allocation.")

    samples_per_class = min_samples_per_client // 2
    num_class_0, num_class_1 = len(class_0_indices), len(class_1_indices)
    class_0_per_client = min(samples_per_class, num_class_0 // num_clients)
    class_1_per_client = min(samples_per_class, num_class_1 // num_clients)

    if class_0_per_client + class_1_per_client < min_samples_per_client:
        shortfall = min_samples_per_client - (class_0_per_client + class_1_per_client)
        if num_class_0 > num_class_1:
            class_0_per_client += min(shortfall, (num_class_0 - class_0_per_client * num_clients) // num_clients)
        else:
            class_1_per_client += min(shortfall, (num_class_1 - class_1_per_client * num_clients) // num_clients)

    if class_0_per_client * num_clients > num_class_0 or class_1_per_client * num_clients > num_class_1:
        raise ValueError("Not enough samples in one or both classes.")

    client_splits = []
    for i in range(num_clients):
        class_0_split = class_0_indices[i * class_0_per_client:(i + 1) * class_0_per_client]
        class_1_split = class_1_indices[i * class_1_per_client:(i + 1) * class_1_per_client]
        client_splits.append(np.concatenate([class_0_split, class_1_split]))

    remaining_class_0 = class_0_indices[class_0_per_client * num_clients:]
    remaining_class_1 = class_1_indices[class_1_per_client * num_clients:]
    remaining_indices = np.concatenate([remaining_class_0, remaining_class_1])
    # prng.shuffle(remaining_indices)
    np.random.shuffle(remaining_indices)
    remaining_samples = len(remaining_indices)

    if remaining_samples > 0:
        # proportions = prng.dirichlet([alpha] * num_clients)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = proportions / proportions.sum()
        split_sizes = (proportions * remaining_samples).astype(int)
        # Cap each client's allocation to less than half of remaining samples
        max_samples_per_client = remaining_samples // 2
        split_sizes = np.minimum(split_sizes, max_samples_per_client)
        # Adjust to ensure all samples are assigned
        total_assigned = split_sizes.sum()
        if total_assigned < remaining_samples:
            deficit = remaining_samples - total_assigned
            # Distribute deficit to clients with room, prioritizing those with fewer samples
            sorted_indices = np.argsort(split_sizes)
            for i in sorted_indices:
                if deficit == 0:
                    break
                additional = min(max_samples_per_client - split_sizes[i], deficit)
                split_sizes[i] += additional
                deficit -= additional
        split_points = np.cumsum(split_sizes)[:-1]
        remaining_client_splits = np.split(remaining_indices, split_points)
        for i in range(num_clients):
            client_splits[i] = np.concatenate([client_splits[i], remaining_client_splits[i]])

    client_dataframes = [data.loc[split].reset_index(drop=True) for split in client_splits]

    total_samples_assigned = sum(len(client_data) for client_data in client_dataframes)
    assert total_samples_assigned == total_samples, f"Sample mismatch: {total_samples_assigned} != {total_samples}"

    return client_dataframes


def iid_distribution(data):
    # Get the label column (last column)
    labels = data.iloc[:, -1]
    # Group data by labels
    grouped = data.groupby(labels)
    
    # Initialize empty lists for each client's data
    client_datasets = [[] for _ in range(num_clients)]
    
    # Split each class's data into num_clients parts
    for _, group in grouped:
        # Split the group's rows into num_clients parts
        splits = np.array_split(group, num_clients)
        # Append each split to the corresponding client's dataset
        for i, split in enumerate(splits):
            client_datasets[i].append(split)
    
    # Concatenate each client's splits into a single DataFrame
    client_datasets = [pd.concat(splits, ignore_index=True) for splits in client_datasets]
    
    # Verify no samples are lost
    total_samples = sum(len(client_data) for client_data in client_datasets)
    assert total_samples == len(data), f"Sample mismatch: {total_samples} != {len(data)}"
    
    return client_datasets


def prepare_client_data(data):
    if non_iid == 1:
        return noniid_distribution(data)
    else:
        return iid_distribution(data)


# Load Dataset Function for loading the dataset and partitioning it into training and testing data
def load_datasets(dataset):
    df = pd.read_csv(f"{dataset_path+dataset}", header=None)
    print(f"Loaded {dataset} with shape: {df.shape}")
    if dataset in continuous_datasets:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

    # Split into train-validation and global test data
    train_val_data, test_data = train_test_split(df, test_size=0.2,random_state=SEED, shuffle=True)
    print(f"Train-Val Data Shape: {train_val_data.shape} | Test Data Shape: {test_data.shape}")

    """Uncomment the following if you want gb before partitioning"""
    gb_making = None
    if make_gb:
        # Convert entire train_val_data to granular balls
        start_time = time.time()
        gb_data = convert_to_granular_balls(train_val_data)
        gb_making = time.time() - start_time
        print(f"Reduced Training Data Size (after granular balls): {(gb_data.shape[0], gb_data.shape[1] - 1)} samples and gb_making time: {gb_making:.2f} seconds")
        time.sleep(1)

    train_data_list = prepare_client_data(train_val_data)
    plot_data_distribution(train_data_list, title=f"Data Distribution for {dataset.split('.')[0]} Dataset")

    gb_train_data_list = None
    if make_gb:
        gb_train_data_list = prepare_client_data(gb_data)
        plot_data_distribution(gb_train_data_list, title=f"Data Distribution for {dataset.split('.')[0]} Dataset (Granular Balls)")



    # """This part after the distribution start making gb"""
    # train_data_list = prepare_client_data(train_val_data)

    # gb_train_data_list = None
    # gb_making = None
    # if make_gb:
    #     start_time = time.time()
    #     gb_train_data_list = [convert_to_granular_balls(client_data) for client_data in train_data_list]
    #     gb_making = time.time() - start_time
    #     print(f"Reduced Training Data Size (after granular balls): {[data.shape for data in gb_train_data_list]} samples and gb_making time: {gb_making:.2f} seconds")
    #     time.sleep(1)

    # plot_data_distribution(train_data_list, title=f"Data Distribution for {dataset.split('.')[0]} Dataset")
    # plot_data_distribution(gb_train_data_list, title=f"Data Distribution for {dataset.split('.')[0]} Dataset (Granular Balls)")

    return train_data_list, test_data, gb_train_data_list, gb_making
