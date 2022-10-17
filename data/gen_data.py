import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from iteration_utilities import random_combination


def generate_shape(bin_num, num_sample=10000):
    mu_range = [0, 1]
    sigma_range = [0.05, 0.5]
    result = []
    config = []
    for _ in range(bin_num):
        mu = random.uniform(mu_range[0], mu_range[1])
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        config.append((mu, sigma))
        s = np.random.normal(mu, sigma, round(num_sample / bin_num)).tolist()
        result.extend(s)
    plt.figure()
    sns.kdeplot(result)
    plt.show()
    return config


def generate_true_dist(config, num_sample=10000, line_style='-', visualize=True):
    result = []
    partial_num = round(num_sample / len(config))
    for (mu, sigma) in config:
        s = np.random.normal(mu, sigma, partial_num).tolist()
        result.extend(s)
    if visualize:
        sns.kdeplot(result, linestyle=line_style)
    return result


def modify_config(config):
    i = np.random.randint(len(config))
    result = config
    for i in random_combination(range(len(config)), r=round(len(config) * 0.5)):
        mu_shift = random.uniform(-0.2, -0.1)
        sigma_shift = random.uniform(0.1, 0.2)
        result = result[:i] + [(result[i][0] + mu_shift, result[i][1] + sigma_shift)] + result[i + 1:]
    return result


def generate_false_dist(config, num_sample=10000, visualize=True):
    return generate_true_dist(modify_config(config), num_sample, line_style='--', visualize=visualize)


def generate_data(num_true, num_false, bin_num, num_sample=10000, visualize=True):
    yes = False
    while not yes:
        config = generate_shape(bin_num, num_sample)
        i = input('Choose this config (y/n): ')
        if i.lower() == 'y':
            yes = True

    data = []
    for _ in range(num_true):
        true = generate_true_dist(config, num_sample, visualize=visualize) + [1]
        data.append(true)
    for _ in range(num_false):
        false = generate_false_dist(config, num_sample, visualize=visualize) + [0]
        data.append(false)
    data = pd.DataFrame(data)
    data.columns = [*data.columns[:-1], 'Label']
    return data