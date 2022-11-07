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


def generate_true_dist(config, num_sample=10000):
    result = []
    partial_num = round(num_sample / len(config))
    for (mu, sigma) in config:
        s = np.random.normal(mu, sigma, partial_num).tolist()
        result.extend(s)
    return result


def modify_config(config):
    i = np.random.randint(len(config))
    result = config
    for i in random_combination(range(len(config)), r=round(len(config) * 0.5)):
        mu_shift = random.uniform(-0.2, -0.1)
        sigma_shift = random.uniform(0.1, 0.2)
        result = result[:i] + [
            (result[i][0] + mu_shift, result[i][1] + sigma_shift)] + result[
                                                                     i + 1:]
    return result


def visualize_data(data, title=''):
    df_false = pd.concat(
        [data[data.Label == 1].iloc[:50], data[data.Label == 0].iloc[:1]])
    df_true = pd.concat(
        [data[data.Label == 1].iloc[:1], data[data.Label == 0].iloc[:50]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    for i, row in df_false.iterrows():
        sns.kdeplot(row, linestyle='--' if row.Label == 1 else '-', ax=ax1)
    for i, row in df_true.iterrows():
        sns.kdeplot(row, linestyle='--' if row.Label == 1 else '-', ax=ax2)

    ax1.set_title('50 false - 1 true')
    ax2.set_title('1 false - 50 true')
    fig.suptitle(f'{title} - Data length: {len(data)}')


def generate_false_dist(bin_num, num_sample=10000):
    return generate_true_dist(generate_shape(bin_num), num_sample)


def generate_data(num_train, num_test, true_ratio, bin_num, num_sample=10000,
                  visualize=True, choose_config=True):
    if choose_config:
        yes = False
        while not yes:
            config = generate_shape(bin_num, num_sample)
            i = input('Choose this config (y/n): ')
            if i.lower() == 'y':
                yes = True
    else:
        config = generate_shape(bin_num, num_sample)
    print('Config:', config)

    def generate(num):
        data = []
        true_num = round(num * true_ratio)
        for _ in range(true_num):
            true = generate_true_dist(config, num_sample) + [0]
            data.append(true)
        for _ in range(num - true_num):
            false = generate_false_dist(bin_num, num_sample) + [1]
            data.append(false)
        random.shuffle(data)
        data = pd.DataFrame(data)
        data.columns = [*data.columns[:-1], 'Label']
        return data

    train, test = generate(num_train), generate(num_test)
    if visualize:
        visualize_data(train, title='Train set')
        visualize_data(test, title='Test set')
    return train, test
