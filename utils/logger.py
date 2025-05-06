import json
import os
from datetime import datetime
# import numpy as np
# import torch


class ResultLogger:
    def __init__(self, experiment_name="experiment"):
        self.experiment_name = experiment_name
        self.results_dir = f"results/{experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)

    def log_generation(self, metrics,):
        """Логирование результатов с временными метками"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        generation = metrics.get('generation', None)
        filename = f"{self.results_dir}/gen_{generation}_{timestamp}.json"
        print(os.path.isdir(self.results_dir))

        result_data = {
            "generation": generation,
            "timestamp": timestamp,
            "metrics": metrics,
            "time_metrics": {
                "training_time": metrics.get('training_time', None),
                "evaluation_time": metrics.get('total_evaluation_time', None),
                "genetic_time": metrics.get('genetic_time', None),
                "total_time": metrics.get('total_time', None),
            },
            "best_individual": {
                "best_ga_loss": metrics.get("best_ga_loss", None),
                "best_ga_accuracy": metrics.get("best_ga_accuracy", None),

                "best_gd_loss": metrics.get("best_gd_loss", None),
                "best_gd_accuracy": metrics.get("best_gd_accuracy", None),
            },
            "population_stats": {
                "mean_ga_loss": metrics.get("mean_ga_loss", None),
                "std_ga_loss": metrics.get("std_ga_loss", None),
                "mean_ga_accuracy": metrics.get("mean_ga_accuracy", None),
                "std_ga_accuracy": metrics.get("std_ga_accuracy", None),

                "mean_gd_loss": metrics.get("mean_gd_loss", None),
                "std_gd_loss": metrics.get("std_gd_loss", None),
                "mean_gd_accuracy": metrics.get("mean_gd_accuracy", None),
                "std_gd_accuracy": metrics.get("std_gd_accuracy", None),
            },
            "ga_config": {
                "generations": metrics.get("generations", None),
                "population_size": metrics.get("population_size", None),
                "crossover_prob": metrics.get("crossover_prob", None),
                "mutation_prob": metrics.get("mutation_prob", None),
            },
        }

        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2)

    def save_final_results(self, history, model=None, config_data=None,):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{self.results_dir}/final_results_{timestamp}.json"

        final_data = {
            "experiment_name": self.experiment_name,
            "timestamp": timestamp,
            "final_metrics": {
                "best_gd_loss": min(history['test_loss_gd']),
                "best_gd_accuracy": max(history['test_accuracy_gd']),
                "best_ga_loss": min(history['test_loss_ga']),
                "best_ga_accuracy": max(history['test_accuracy_ga']),

                "total_training_time": sum(
                    t['train'] for t in history['time']),
                "total_evaluation_time": sum(
                    t['evaluation_gd'] + t["evaluation_ga"]
                    for t in history['time'])
            },
            "training_history": history
        }

        if config_data is not None:
            final_data["ga_config"] = {
                'generations': config_data.get('generations', None),
                'population_size': config_data.get('population_size', None),
                'crossover_prob': config_data.get('crossover_prob', None),
                'mutation_prob': config_data.get('mutation_prob', None),
            }

        if model is not None:
            final_data["model_architecture"] = str(model)

        with open(filename, 'w') as f:
            json.dump(final_data, f, indent=2)

        return final_data
