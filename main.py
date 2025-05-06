# import sys
# sys.path.append("..")
# sys.path.append(".")
import time
import random
import numpy as np
import torch
import torch.nn as nn
from deap import creator  # base,
from networks.train import train_generation
from config import (
    SEED,
    DEVICE,
    GENERATIONS,
    POPULATION_SIZE,
    CROSSOVER_PROB,
    MUTATION_PROB,
)
from utils.dataloader import load_mnist
from genetic.population import Population
from genetic.operators import setup_genetic_operators
import logging
from utils.logger import ResultLogger


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filemode="a",
    filename="train_ga_gd_log.log",
)


def main():
    # Initialisation
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Data Download
    train_loader, test_loader = load_mnist()

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Population Initialisation
    population = Population(POPULATION_SIZE)
    toolbox = setup_genetic_operators()

    # history_gd = {'test_loss': [], 'test_accuracy': [], 'time': []}
    # history_ga = {'test_loss': [], 'test_accuracy': [], 'time': []}
    history = {
        'test_loss_gd': [],
        'test_accuracy_gd': [],
        'test_loss_ga': [],
        'test_accuracy_ga': [],
        'time': [],
    }
    custom_logger = ResultLogger("mnist_hybrid_ga/exp_16_cx_08_mut_015")
    for gen in range(GENERATIONS):
        logger.info(f"--- Generation {gen + 1} ---")
        gen_start = time.time()

        # Gradient descent phase
        train_start = time.time()
        train_generation(population, train_loader, loss_fn, DEVICE)
        train_time = time.time() - train_start

        # Evaluation
        eval_gd_start = time.time()
        population.evaluate_generation(test_loader, loss_fn, DEVICE)
        eval_gd_time = time.time() - eval_gd_start

        # Write best scores after gradient descent
        best_gd_loss, best_gd_acc = np.min(population.fitness[:, 0]), np.max(population.fitness[:, 1])
        mean_gd_loss, mean_gd_accuracy = np.mean(population.fitness[:, 0]), np.mean(population.fitness[:, 1])
        std_gd_loss, std_gd_accuracy = np.std(population.fitness[:, 0]), np.std(population.fitness[:, 1])
        history["test_loss_gd"].append(best_gd_loss)
        history["test_accuracy_gd"].append(best_gd_acc)
        logger.info(f"Best Loss: {best_gd_loss:.4f}, Accuracy: {best_gd_acc:.2%}")

        # Genetic phase
        logging.info("GA phase started")
        ga_start = time.time()
        params = population.get_parameters()
        population_deap = [creator.Individual(p) for p in params]

        # calculate fitness tuple for each individual in the population:
        logging.info("Calculate fitness tuple for each individual in the population")
        for ind, (loss, acc) in zip(population_deap, population.fitness):
            ind.fitness.values = (loss, acc)

        logging.info("Apply the selection operator, to select the next generation's individuals")
        # apply the selection operator, to select the next generation's individuals:
        offspring = toolbox.select(population_deap, len(population_deap))

        # clone the selected individuals:
        offspring = list(map(toolbox.clone, offspring))

        # apply the crossover operator to pairs of offspring:
        logging.info("Apply the crossover operator to pairs of offspring")
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROB:  # Вероятность кроссовера
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # replace the current population with the offspring:
        logging.info("Replace the current population with the offspring")
        population.set_parameters(offspring)
        ga_time = time.time() - ga_start

        # Evaluation
        eval_ga_start = time.time()
        population.evaluate_generation(test_loader, loss_fn, DEVICE)
        eval_ga_time = time.time() - eval_ga_start

        # Write best scores after genetic algorithm
        best_ga_loss, best_ga_acc = np.min(population.fitness[:, 0]), np.max(population.fitness[:, 1])
        mean_ga_loss, mean_ga_accuracy = np.mean(population.fitness[:, 0]), np.mean(population.fitness[:, 1])
        std_ga_loss, std_ga_accuracy = np.std(population.fitness[:, 0]), np.std(population.fitness[:, 1])

        history["test_loss_ga"].append(best_ga_loss)
        history["test_accuracy_ga"].append(best_ga_acc)
        logger.info(f"Best Loss: {best_ga_loss:.4f}, Accuracy: {best_ga_acc:.2%}")

        gen_time = time.time() - gen_start
        time_metrics = {
            'train': train_time,
            'evaluation_gd': eval_gd_time,
            'genetic': ga_time,
            'evaluation_ga': eval_ga_time,
            'total': gen_time
        }
        history['time'].append(time_metrics)

        # Логирование
        custom_logger.log_generation(
            metrics={
                'phase': 'full_cycle',
                'generation': gen,
                'best_ga_loss': best_ga_loss,
                'best_ga_accuracy': best_ga_acc,
                'best_gd_loss': best_gd_loss,
                'best_gd_accuracy': best_gd_acc,
                'training_time': train_time,
                'genetic_time': ga_time,
                'eval_gd_time': eval_gd_time,
                'eval_ga_time': eval_ga_time,
                'total_evaluation_time': eval_gd_time + eval_ga_time,
                'total_time': gen_time,

                "mean_gd_loss": mean_gd_loss,
                "std_gd_loss": std_gd_loss,
                "mean_gd_accuracy": mean_gd_accuracy,
                "std_gd_accuracy": std_gd_accuracy,

                "mean_ga_loss": mean_ga_loss,
                "std_ga_loss": std_ga_loss,
                "mean_ga_accuracy": mean_ga_accuracy,
                "std_ga_accuracy": std_ga_accuracy,

                'generations': GENERATIONS,
                'population_size': POPULATION_SIZE,
                'crossover_prob': CROSSOVER_PROB,
                'mutation_prob': MUTATION_PROB,
            },
            # population=population,
        )

    best_model = population.get_best_model()
    custom_logger.save_final_results(
        history,
        best_model,
        config_data={
            'generations': GENERATIONS,
            'population_size': POPULATION_SIZE,
            'crossover_prob': CROSSOVER_PROB,
            'mutation_prob': MUTATION_PROB,
        }
    )


if __name__ == "__main__":
    main()
