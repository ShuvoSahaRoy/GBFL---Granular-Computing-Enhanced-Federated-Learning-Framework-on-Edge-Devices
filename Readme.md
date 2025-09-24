
# Granular Computing Enhanced FederatedLearning Framework on Edge Devices

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed edge clients such as mobile or IoT devices by sharing model updates rather than raw data, preserving privacy while supporting scalable learning. However, large datasets increase computational costs and are time-consuming, hindering efficient training, especially in edge scenarios with limited compute and bandwidth. We propose Granular Computing (GC)-powered federated learning where granular balls are computed as a representative of clustered data, providing a more structured way to represent and process data for each client, leading to improved efficiency of machine learning models. The proposed framework reduces training data by 95.5\%, significantly accelerating simulation times (76–96\% reduction) and enabling low-latency, resource-efficient processing at the edge across simple (logistic regression) and complex (NN) models using FedAvg. Moreover, by sharing only granular characteristics, GC theoretically enhances privacy. Our results highlight GC’s effectiveness for resource-constrained edge environments and large-scale FL simulations at the edge, enabling faster experimentation with only a negligible 1–3\% accuracy trade-off, preserving comparable performance for real-time edge applications.


## Authors

- [@Shuvo Saha Roy](https://github.com/ShuvoSahaRoy/)
- [@Dr Reshma Rastogi](https://scholar.google.com/citations?user=NqIkygEAAAAJ&hl=en)

## Installation

Set up the environment, check the config file, change values according to your preference, and then run the main file. I have used Python v3.11

```bash
python -m vevn GBFL
pip install -r requirements.txt
```

Activate the environment and just run the main.py file.

```bash
python main.py
``` 
## License

[MIT](https://choosealicense.com/licenses/mit/)

