import pandas as pd
import numpy as np

population_stats = {
    ((15, 19), 'F'): (1850, 343.334),
    ((20, 24), 'F'): (1880, 312.122),
    ((25, 29), 'F'): (1920, 323.046),
    ((30, 34), 'F'): (1960, 335.531),
    ((35, 39), 'F'): (1990, 391.713),
    ((40, 44), 'F'): (2070, 388.592),
    ((45, 49), 'F'): (2530, 380.669),
    ((50, 54), 'F'): (2880, 400.548),
    ((55, 59), 'F'): (3000, 400.093),
    ((60, 64), 'F'): (3540, 234.092),
    ((65, 69), 'F'): (3600, 234.092),
    ((15, 19), 'M'): (1440, 269.205),
    ((20, 24), 'M'): (1530, 327.728),
    ((25, 29), 'M'): (1590, 280.909),
    ((30, 34), 'M'): (1655, 312.122),
    ((35, 39), 'M'): (1675, 312.122),
    ((40, 44), 'M'): (1695, 327.728),
    ((45, 49), 'M'): (1740, 339.432),
    ((50, 54), 'M'): (1980, 480.667),
    ((55, 59), 'M'): (2000, 411.22),
    ((60, 64), 'M'): (2200, 475.986),
    ((65, 69), 'M'): (2200, 312.122),
}

def find_age_range(age, gender):
    for (age_range, g), (mean, std) in population_stats.items():
        if age_range[0] <= age <= age_range[1] and g == gender:
            return mean, std
    return None, None

def calculate_fitness_score(predicted_5k_time, age, gender):
    mean_5k_time, std_5k_time = find_age_range(age, gender)
    if mean_5k_time is None or std_5k_time is None:
        raise ValueError("No population statistics available for the specified age and gender group.")
    fitness_score = 50 + 10 * (mean_5k_time - predicted_5k_time) / std_5k_time
    return fitness_score
