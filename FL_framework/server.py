from collections import defaultdict
import flwr as fl
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Storage for all metrics
training_metrics = defaultdict(lambda: defaultdict(list))   

eval_metrics = defaultdict(lambda: defaultdict(list))      

def start_server(noise_levels=[1.0]):
    for noise in noise_levels:
        print(f"\nStarting server with noise level: {noise}")

        class DPFedAvg(fl.server.strategy.FedAvg):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def aggregate_fit(self, rnd, results, failures):
                if failures:
                    return None, {}

                aggregated_result, _ = super().aggregate_fit(rnd, results, failures)
                if aggregated_result is None:
                    return None, {}

                total_examples = sum(fit_res.num_examples for _, fit_res in results)

                accuracy = sum(fit_res.metrics["accuracy"] * fit_res.num_examples for _, fit_res in results if "accuracy" in fit_res.metrics) / total_examples
                precision = sum(fit_res.metrics["precision"] * fit_res.num_examples for _, fit_res in results if "precision" in fit_res.metrics) / total_examples
                recall = sum(fit_res.metrics["recall"] * fit_res.num_examples for _, fit_res in results if "recall" in fit_res.metrics) / total_examples
                epsilons = [fit_res.metrics.get("epsilon", 0.0) for _, fit_res in results]
                avg_epsilon = sum(epsilons) / len(epsilons) if epsilons else 0.0

                # Store training metrics
                training_metrics[rnd]["accuracy"].append(accuracy)
                training_metrics[rnd]["precision"].append(precision)
                training_metrics[rnd]["recall"].append(recall)
                training_metrics[rnd]["epsilon"].append(avg_epsilon)

                print(f"[Round {rnd}] Training - Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, ε: {avg_epsilon:.2f}")
                return aggregated_result, {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "epsilon": avg_epsilon
                }

            def aggregate_evaluate(self, rnd, results, failures):
                if failures:
                    return None, {}

                total_examples = sum(eval_res.num_examples for _, eval_res in results)
                loss = sum(eval_res.loss * eval_res.num_examples for _, eval_res in results) / total_examples
                accuracy = sum(eval_res.metrics["accuracy"] * eval_res.num_examples for _, eval_res in results if "accuracy" in eval_res.metrics) / total_examples
                precision = sum(eval_res.metrics["precision"] * eval_res.num_examples for _, eval_res in results if "precision" in eval_res.metrics) / total_examples
                recall = sum(eval_res.metrics["recall"] * eval_res.num_examples for _, eval_res in results if "recall" in eval_res.metrics) / total_examples

                # Store evaluation metrics
                eval_metrics[rnd]["loss"].append(loss)
                eval_metrics[rnd]["accuracy"].append(accuracy)
                eval_metrics[rnd]["precision"].append(precision)
                eval_metrics[rnd]["recall"].append(recall)

                print(f"[Eval {rnd}] Eval - Loss: {loss:.4f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")
                return loss, {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                }

        fl.server.start_server(
            server_address="127.0.0.1:8080",
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=DPFedAvg(
                on_fit_config_fn=lambda rnd: {
                    "batch_size": 2,
                    "local_epochs": 1,
                    "noise_multiplier": 1.0
                },
                fit_metrics_aggregation_fn=lambda metrics: {}  # <- suppress Flower's warning
            )
        )

def main():
    start_server()


if __name__ == "__main__":
    main()


    # After running all noise levels
    print("\n==== FINAL TRAINING METRICS ====")
    for noise, metrics in training_metrics.items():
        print(f"\nRound {noise}")
        for k, v in metrics.items():
            print(f"Train {k.capitalize()}: {v}")

    print("\n==== FINAL EVALUATION METRICS ====")
    for noise, metrics in eval_metrics.items():
        print(f"\nRound {noise}")
        for k, v in metrics.items():
            print(f"Eval {k.capitalize()}: {v}")
