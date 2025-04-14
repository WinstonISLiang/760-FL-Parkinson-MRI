import flwr as fl
from flwr.server.strategy import FedAvg
import numpy as np

#  average result of three client accuracy
def aggregate_metrics(metrics):
    accuracies = [m["accuracy"] for _, m in metrics]
    mean_acc = float(np.mean(accuracies))
    print(f"[Server] Round accuracy (avg across clients): {mean_acc:.2%}", flush=True)
    return {"accuracy": mean_acc}

strategy = FedAvg(
    evaluate_metrics_aggregation_fn=aggregate_metrics
)

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy
)