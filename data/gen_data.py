import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from iteration_utilities import random_combination


def gen_nextday(prev, true):
    '''
        Mix with a random normal distribution in some ratio to form new day with respect to the previous day
        - prev: python array, all the values of previous day
        - r: ratio of mixing with another
    '''
    mu_range = [0, 1]
    sigma_range = [0.3, 0.5]
    if true:
        r=0.05
        sigma_range = [0.3, 0.5]
    else:
        r = 0.3
        sigma_range = [0.05, 0.2]
    # remove random r% of element in prev
    result = random.sample(prev, round(len(prev) * (1 - r)))
    # add r% of a new random normal distribution to result
    mu = random.uniform(mu_range[0], mu_range[1])
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    s = np.random.normal(mu, sigma, len(prev)).tolist()
    result = result + random.sample(s, round(len(prev) * r))
    # shuffle result and return
    random.shuffle(result)
    return result


def generate_data():
    num_days = 700
    num_sample = 10000

    bin_num = 5
    mu_range = [0, 1]
    sigma_range = [0.05, 0.5]

    data = pd.DataFrame(columns=list(range(num_sample)) + ['Labels'])

    # generate first day
    first_day = []
    for _ in range(bin_num):
        mu = random.uniform(mu_range[0], mu_range[1])
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        s = np.random.normal(mu, sigma, round(num_sample / bin_num)).tolist()
        first_day.extend(s)

    data.loc[0] = first_day + [1]

    # 1 is true, 0 is false

    prev = first_day
    for day in range(1, num_days):
        label = np.random.choice([1, 0], p=[0.7, 0.3])
        if (label == 1):
            next = gen_nextday(prev, True)
        else:
            next = gen_nextday(prev, False)
        prev = next
        data.loc[day] = next + [label]

    plt.figure()
    for i,row in data.loc[0:5].iterrows():
        sns.kdeplot(row.drop('Labels'), color= 'red' if row['Labels'] == 0 else 'blue', multiple='stack')
    plt.show()
    print(data.loc[0:5])

    return data
