import torch

# Общие настройки
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Настройки генетического алгоритма
POPULATION_SIZE = 150
GENERATIONS = 10
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.15

# Настройки обучения
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRAINING_STEPS = 2