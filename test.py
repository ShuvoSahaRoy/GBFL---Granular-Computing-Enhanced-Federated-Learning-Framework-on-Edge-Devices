# import numpy as np


# class SVM:
#     def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
#         self.lr = learning_rate
#         self.lambda_param = lambda_param
#         self.n_iters = n_iters
#         self.w = None
#         self.b = None

#     def fit(self, X, y):
#         n_samples, n_features = X.shape

#         y_ = np.where(y <= 0, -1, 1)

#         self.w = np.zeros(n_features)
#         self.b = 0

#         for _ in range(self.n_iters):
#             for idx, x_i in enumerate(X):
#                 condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
#                 if condition:
#                     self.w -= self.lr * (2 * self.lambda_param * self.w)
#                 else:
#                     self.w -= self.lr * (
#                         2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
#                     )
#                     self.b -= self.lr * y_[idx]

#     def predict(self, X):
#         approx = np.dot(X, self.w) - self.b
#         return np.sign(approx)


# # Testing
# if __name__ == "__main__":
#     # Imports
#     from sklearn import datasets
#     import matplotlib.pyplot as plt

#     X, y = datasets.make_blobs(
#         n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
#     )
#     y = np.where(y == 0, -1, 1)

#     clf = SVM()
#     clf.fit(X, y)
#     # predictions = clf.predict(X)

#     print(clf.w, clf.b)

#     def visualize_svm():
#         def get_hyperplane_value(x, w, b, offset):
#             return (-w[0] * x + b + offset) / w[1]

#         fig = plt.figure()
#         ax = fig.add_subplot(1, 1, 1)
#         plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

#         x0_1 = np.amin(X[:, 0])
#         x0_2 = np.amax(X[:, 0])

#         x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
#         x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

#         x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
#         x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

#         x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
#         x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

#         ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
#         ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
#         ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

#         x1_min = np.amin(X[:, 1])
#         x1_max = np.amax(X[:, 1])
#         ax.set_ylim([x1_min - 3, x1_max + 3])

#         plt.show()

#     visualize_svm()


# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# datasets = ['QSAR Oral Toxicity', 'Phishing', '2D Planes', 'A9a', 'Adult', 'Hepmass']
# original = [7193, 8844, 32614, 39073, 39073, 262021]
# granular = [269, 314, 680, 692, 692, 7768]

# x = np.arange(len(datasets))  # Label locations
# width = 0.35  # Bar width

# # Plot
# fig, ax = plt.subplots(figsize=(10, 6))
# bars1 = ax.bar(x - width/2, original, width, label='Original Datapoints', color='steelblue')
# bars2 = ax.bar(x + width/2, granular, width, label='Granular Balls', color='darkorange')

# # Log scale for Y-axis
# ax.set_yscale('log')

# # Labels & titles
# ax.set_ylabel('Count (log scale)', fontsize=18)
# ax.set_xlabel('Dataset', fontsize=18)
# # ax.set_title('Comparison of Original Datapoints and Granular Balls', fontsize=14)
# ax.set_xticks(x)
# ax.set_xticklabels(datasets)
# ax.legend()
# ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)

# # Optional: Add value labels
# def add_labels(bars):
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 5),  # Offset text
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=8)

# add_labels(bars1)
# add_labels(bars2)

# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# plt.rcParams.update({
#     'font.size': 22,
#     'axes.titlesize': 22,
#     'axes.labelsize': 22,
#     'xtick.labelsize': 22,
#     'ytick.labelsize': 22,
#     'legend.fontsize': 24
# })

# # Data
# datasets = ['QOT', 'Phishing', '2D Planes', 'A9a', 'Adult', 'Hepmass']
# original = [7193, 8844, 32614, 39073, 39073, 262021]
# granular = [269, 314, 680, 692, 692, 7768]

# x = np.arange(len(datasets))  # Label locations
# width = 0.35  # Bar width

# fig, ax = plt.subplots(figsize=(10, 6))
# bars1 = ax.bar(x - width/2, original, width, label='Original Datapoints', color='steelblue')
# bars2 = ax.bar(x + width/2, granular, width, label='Granular Balls', color='darkorange')

# ax.set_yscale('log')
# ax.set_ylabel('Count (log scale)')
# ax.set_xlabel('Dataset')
# ax.set_xticks(x)
# ax.set_xticklabels(datasets)
# ax.legend()
# ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)

# def add_labels(bars):
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 5),
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=16)  # <-- font size here

# add_labels(bars1)
# add_labels(bars2)

# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data structure: {dataset: [(algorithm, iid_time, noniid_time)]}
data = {
'A9a': [
        ('M1', 2.41, 2.40),
        ('GB_M1', 0.37, 0.50),
        ('M2', 126.10, 103.99),
        ('GB_M2', 5.11, 4.19)
    ],


'Adult': [
        ('M1', 2.63, 2.27),
        ('GB_M1', 0.32, 0.41),
        ('M2', 126.94, 121.36),
        ('GB_M2', 4.88, 4.53)
    ],

    '2D Planes': [
        ('M1', 0.32, 0.29),
        ('GB_M1', 0.15, 0.14),
        ('M2', 98.69, 91.19),
        ('GB_M2', 4.75, 4.13)
    ],
    'Hepmass': [
        ('M1', 1.79, 2.07),
        ('GB_M1', 1.03, 0.95),
        ('M2', 800.72, 740.06),
        ('GB_M2', 29.48, 28.79)
    ],
    'Phishing': [
        ('M1', 0.23, 0.17),
        ('GB_M1', 0.10, 0.10),
        ('M2', 30.07, 33.15),
        ('GB_M2', 2.44, 2.45)
    ],
    'QSAR Oral Toxicity': [
        ('M1', 3.53, 3.77),
        ('GB_M1', 0.56, 0.56),
        ('M2', 38.74, 43.11),
        ('GB_M2', 2.62, 2.50)
    ],
}

# Algorithm pairs to compare
pairs = [('M1', 'GB_M1'), ('M2', 'GB_M2')]

# Appearance settings
bar_width = 0.4
colors = ['b', 'c']  # IID, Non-IID
y_max_margin = 2  # Adjusted for square root scale

for dataset, times in data.items():
    algorithms = [x[0] for x in times]
    iid_times = [x[1] for x in times]
    noniid_times = [x[2] for x in times]

    # Apply square root scaling for plotting only
    iid_times_scaled = [np.sqrt(x) for x in iid_times]
    noniid_times_scaled = [np.sqrt(x) for x in noniid_times]

    x = np.arange(len(algorithms))
    max_time_scaled = max(max(iid_times_scaled), max(noniid_times_scaled))

    plt.figure(figsize=(8, 6))

    # Plot bars using scaled values
    iid_bars = plt.bar(x - bar_width/2, iid_times_scaled, width=bar_width, label='IID', color=colors[0])
    noniid_bars = plt.bar(x + bar_width/2, noniid_times_scaled, width=bar_width, label='Non-IID', color=colors[1])

    # Annotate bars with real values (not scaled)
    for i in range(len(algorithms)):
        plt.text(x[i] - bar_width/2, iid_times_scaled[i] + 0.1, f"{iid_times[i]:.2f}", ha='center', fontsize=14)
        plt.text(x[i] + bar_width/2, noniid_times_scaled[i] + 0.1, f"{noniid_times[i]:.2f}", ha='center', fontsize=14)

    plt.legend(loc='upper left', fontsize=22)
    plt.xticks(x, algorithms, fontsize=22)
    plt.ylabel(f"Execution Time (√s)\nSquare Root Scaled", fontsize=22)
    # plt.title(f"Execution Time by Algorithm: {dataset}")
    plt.ylim(0, max_time_scaled + y_max_margin)
    plt.tight_layout()
    plt.savefig(f"barplot_{dataset}.png", dpi=300, bbox_inches='tight')
    # plt.show()