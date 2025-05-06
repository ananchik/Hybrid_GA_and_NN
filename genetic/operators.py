import random
import numpy as np
from deap import base, creator, tools


def setup_genetic_operators():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0))  # Минимизируем loss, максимизируем accuracy
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    # toolbox.register("mate", tools.cxSimulatedBinary, eta=30)
    # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.1)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=15, indpb=0.1, low=-1., up=1.,)

    return toolbox