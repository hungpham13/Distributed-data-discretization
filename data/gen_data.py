import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from model.psi import get_breakpoint, calculate_psi
import gc


def generate_day(length, dist, mu_range, sigma_range, value_range):
    mu = random.uniform(mu_range[0], mu_range[1])

    max_sigma = min(mu-value_range[0], value_range[1]-mu)/5
    min_sigma = sigma_range[0] if sigma_range[0] < max_sigma else max_sigma
    sigma = random.uniform(min_sigma, max_sigma)

    if dist == 'mix':
        d = np.random.choice(
            ['uniform', 'logistic', 'normal', 'exponential', 'gamma'])
    else:
        d = dist

    if (d == 'uniform'):
        s = np.random.uniform(mu-3*3*sigma/2,
                              mu+3*3*sigma/2, length).tolist()
    elif (d == 'logistic'):
        s = np.random.logistic(mu, sigma/np.pi, length).tolist()
    elif (d == 'exponential'):
        # sigma = random.uniform(0, 85)
        s = (np.random.exponential(sigma/1.47, length) + mu-sigma*5).tolist()
    elif (d == 'gamma'):
        s = np.random.gamma((mu/sigma)**2, sigma**2/mu, length).tolist()
    else:
        s = np.random.normal(mu, sigma, length).tolist()
    return [round(i) for i in s]


def gen_nextday(prev, true, dist):
    '''
        Mix with a random normal distribution in some ratio to form new day with respect to the previous day
        - prev: python array, all the values of previous day
        - r: ratio of mixing with another
    '''
    value_range = [300, 850]
    mu_range = [525, 625]
    if true:
        r = 0.1
        sigma_range = [90,]
    else:
        r = 0.9
        sigma_range = [45, ]
    # remove random r% of element in prev
    result = random.sample(prev, round(len(prev) * (1 - r)))
    # add r% of a new random normal distribution to result
    s = generate_day(len(prev), dist, mu_range, sigma_range, value_range)
    result = result + random.sample(s, round(len(prev) * r))
    # shuffle result and return
    random.shuffle(result)
    return result


def generate_data(num_days, num_samples, dist):
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

    print(
        f'Generating {dist} distribution, {num_days} days, {num_samples} samples...')
    # generate first day
    first_day = []
    for _ in range(bin_num):
        s = generate_day(num_samples, dist, mu_range, sigma_range, value_range)
        first_day.extend(s)

    data = np.array([first_day + [0]])

    # 0 is true, 1 is false

    prev = first_day
    breakpoints = get_breakpoint(first_day, buckettype='bins', buckets=10)
    for day in tqdm(range(1, num_days)):
        label = np.random.choice([0, 1], p=[0.7, 0.3])
        if (label == 0):
            next = gen_nextday(prev, True, dist)
        else:
            next = gen_nextday(prev, False, dist)
            psi = calculate_psi(expected=np.array(prev), actual=np.array(next),
                                breakpoints=breakpoints)
            while (psi < 0.1):
                next = gen_nextday(prev, False, dist)
                psi = calculate_psi(expected=np.array(prev), actual=np.array(next),
                                    breakpoints=breakpoints)

        prev = next
        data = np.append(data, [next + [label]], axis=0)
        gc.collect()

    plt.figure()
    for row in data[0:6, :]:
        sns.kdeplot(row[:-1], color='blue' if row[-1]
                    == 0 else 'red', multiple='stack')
    plt.show()
    print(data[:5, :])

    return data
