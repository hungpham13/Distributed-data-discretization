import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from iteration_utilities import random_combination


def gen_nextday(prev, true, dist):
    '''
        Mix with a random normal distribution in some ratio to form new day with respect to the previous day
        - prev: python array, all the values of previous day
        - r: ratio of mixing with another
    '''
    value_range = [300, 850]
    mu_range = [525, 625]
    if true:
        r = 0.05
        sigma_range = [90,]
    else:
        r = 0.3
        sigma_range = [45, ]
    # remove random r% of element in prev
    result = random.sample(prev, round(len(prev) * (1 - r)))
    # add r% of a new random normal distribution to result
    mu = random.uniform(mu_range[0], mu_range[1])

    max_sigma = max(
        min(mu-value_range[0], value_range[1]-mu)/6, sigma_range[0])
    sigma = random.uniform(sigma_range[0], max_sigma)

    if dist == 'mix':
        d = np.random.choice(['uniform', 'logistic', 'normal'])
    else:
        d = dist

    if (d == 'uniform'):
        s = np.random.uniform(mu-3*np.sqrt(12)*sigma/2,
                              mu+3*np.sqrt(12)*sigma/2, len(prev)).tolist()
    elif (d == 'logistic'):
        s = np.random.logistic(mu, sigma*np.sqrt(3)/np.pi, len(prev)).tolist()
    else:
        s = np.random.normal(mu, sigma, len(prev)).tolist()
    s = [round(i) for i in s]
    result = result + random.sample(s, round(len(prev) * r))
    # shuffle result and return
    random.shuffle(result)
    return result


def generate_data(num_days, num_sample, dist):
    '''
        num_days: number,
        num_sample: number,
        dist: 'normal' or 'logistic' or 'uniform' or 'mix',
    '''
    # num_days = 700
    # num_sample = 10000

    bin_num = 1
    value_range = [300, 850]
    mu_range = [450, 700]
    sigma_range = [45, ]

    data = pd.DataFrame(columns=list(range(num_sample)) + ['Labels'])

    # generate first day
    first_day = []
    for _ in range(bin_num):
        mu = random.uniform(mu_range[0], mu_range[1])
        max_sigma = max(
            min(mu-value_range[0], value_range[1]-mu)/6, sigma_range[0])
        sigma = random.uniform(sigma_range[0], max_sigma)
        if dist == 'mix':
            d = np.random.choice(['uniform', 'logistic', 'normal'])
        else:
            d = dist
        if (d == 'uniform'):
            s = np.random.uniform(mu-3*np.sqrt(12)/2*sigma, mu+3*np.sqrt(12)*sigma/2,
                                  round(num_sample / bin_num)).tolist()
        elif (d == 'logistic'):
            s = np.random.logistic(mu, sigma*np.sqrt(3)/np.pi, round(
                num_sample / bin_num)).tolist()
        else:
            s = np.random.normal(mu, sigma, round(
                num_sample / bin_num)).tolist()
        first_day.extend([round(i) for i in s])

    data.loc[0] = first_day + [0]

    # 0 is true, 1 is false

    prev = first_day
    for day in range(1, num_days):
        label = np.random.choice([0, 1], p=[0.7, 0.3])
        if (label == 0):
            next = gen_nextday(prev, True, dist)
        else:
            next = gen_nextday(prev, False, dist)
        prev = next
        data.loc[day] = next + [label]

    plt.figure()
    for i, row in data.loc[0:5].iterrows():
        sns.kdeplot(row.drop(
            'Labels'), color='blue' if row['Labels'] == 0 else 'red', multiple='stack')
    plt.show()
    print(data.loc[0:5])

    return data
