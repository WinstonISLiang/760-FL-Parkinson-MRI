import flwr as fl
from collections import defaultdict
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Storage for all metrics
eval_metrics = defaultdict(lambda: defaultdict(list))

def start_server():
    class DPFedAvg(fl.server.strategy.FedAvg):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def aggregate_fit(self, rnd, results, failures):
            if failures:
                return None, {}

            aggregated_result, _ = super().aggregate_fit(rnd, results, failures)
            print(f"[Round {rnd}] Aggregated model updates with simulated Secure Aggregation.")
            return aggregated_result, {}  # No client-level metrics returned

        def aggregate_evaluate(self, rnd, results, failures):
            if failures:
                return None, {}

            total_examples = sum(eval_res.num_examples for _, eval_res in results)
            loss = sum(eval_res.loss * eval_res.num_examples for _, eval_res in results) / total_examples
            accuracy = sum(eval_res.metrics["accuracy"] * eval_res.num_examples for _, eval_res in results if "accuracy" in eval_res.metrics) / total_examples
            precision = sum(eval_res.metrics["precision"] * eval_res.num_examples for _, eval_res in results if "precision" in eval_res.metrics) / total_examples
            recall = sum(eval_res.metrics["recall"] * eval_res.num_examples for _, eval_res in results if "recall" in eval_res.metrics) / total_examples

            eval_metrics[rnd]["loss"].append(loss)
            eval_metrics[rnd]["accuracy"].append(accuracy)
            eval_metrics[rnd]["precision"].append(precision)
            eval_metrics[rnd]["recall"].append(recall)

            print(f"[Eval {rnd}] Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            return loss, {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
            }

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=DPFedAvg(
            on_fit_config_fn=lambda rnd: {
                "batch_size": 2,
                "local_epochs": 1,
                "noise_multiplier": 1.0
            },
            fit_metrics_aggregation_fn=lambda metrics: {}  # Suppress unused
        )
    )

if __name__ == "__main__":
    start_server()
