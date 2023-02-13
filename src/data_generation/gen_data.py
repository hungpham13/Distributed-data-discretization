import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from model.utils import js, kl_div, psi, wd
from scipy.ndimage.filters import gaussian_filter1d
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
    
    result = np.array([round(i) for i in s])
    result[result > 850] = 850
    result[result < 300] = 300

    return result.tolist()


def gen_nextday(prev, true, dist):
    '''
        Mix with a random normal distribution in some ratio to form new day with respect to the previous day
        - prev: python array, all the values of previous day
        - r: ratio of mixing with another
    '''
    value_range = [300, 850]
    mu_range = [600, 700]
    sigma_range = [100,]
    if true:
        r = 0.1
    else:
        r = 0.6
    # remove random r% of element in prev
    result = random.sample(prev, round(len(prev) * (1 - r)))
    # add r% of a new random normal distribution to result
    s = generate_day(len(prev), dist, mu_range, sigma_range, value_range)
    result = result + random.sample(s, round(len(prev) * r))
    # shuffle result and return
    # random.shuffle(result)
    return result


def add_drift(reference, drift_size: float, drift_ratio: float, drift_mode: str='fixed'):
    """Artificially adds a shift to the data.
    Args:
    curr: initial data
    drift_size: percents initial values would be increased by
    drift_ratio: defines what percent of data would be drifted
    drift_mode:
        if drift_mode == 'fixed': 
        # here we should use mean(reference), but in out experiment mean(reference) = mean(current) at this stage
        all values moved by fixed delta = (alpha + mean(feature)) * drift_size
        elif: 
        drift_mode == 'relative': vlues moved by delta(value) = value * drift_size
    Returns:
    curr: drifted data
    """
    alpha = 0.001
    i = int(np.round(len(reference) * drift_ratio))
    change_sign = np.random.choice([1, 0, -1], p=[0.3, 0.5, 0.2])
    # change_sign = 1
    if drift_mode == 'fixed':
        delta = int((alpha + np.mean(reference)) * drift_size)
        reference[:i] = reference[:i] + change_sign*delta
    else:
        reference = reference.astype(float)
        reference[:i] = reference[:i]*(1 + change_sign*drift_size)
        reference = reference.astype(int)
    reference[reference > 850] = 850
    reference[reference < 300] = 300

    return reference.tolist()


def generate(num_days, num_samples, dist, mode, normal_ratio=0.9):
    bin_num = 1
    value_range = [300, 850]
    mu_range = [600, 700]
    sigma_range = [100, ]

    def hist(arr):
        h, _ = np.histogram(arr, bins=np.arange(
            value_range[0], value_range[1] + 1, 1))
        return h.tolist()

    print(
        f'Generating {dist} distribution, {num_days} days, {num_samples} samples...')
    # generate first day
    first_day = []
    for _ in range(bin_num):
        s = generate_day(num_samples, dist, mu_range, sigma_range, value_range)
        first_day.extend(s)

    if (mode == 'histogram'):
        data = np.array([hist(first_day) + [0]])
    else:
        data = np.array([first_day + [0]])

    # 0 is true, 1 is false

    prev = first_day
    for day in tqdm(range(1, num_days)):
        label = np.random.choice([0, 1], p=[normal_ratio, 1 - normal_ratio])

        if (label == 0):
            next = gen_nextday(prev, True, dist)

            count = 0
            thresholds = np.random.normal(0.1, 0.001, 4)
            thresholds[thresholds < 0.05] = 0.05
            thresholds[thresholds > 0.3] = 0.3

            scores = [js(prev, next, threshold=thresholds[0])[0],
                    kl_div(prev, next, threshold=thresholds[1])[0],
                    psi(prev, next, threshold=thresholds[2])[0],
                    wd(prev, next, threshold=thresholds[3])[0]]
            
            votes = [js(prev, next, threshold=thresholds[0])[1],
                    kl_div(prev, next, threshold=thresholds[1])[1],
                    psi(prev, next, threshold=thresholds[2])[1],
                    wd(prev, next, threshold=thresholds[3])[1]]

            stop = np.mean(scores) < 0.1 and sum(votes) <= 1
            while not stop:
                next = gen_nextday(prev, False, dist)

                scores = [js(prev, next, threshold=thresholds[0])[0],
                        kl_div(prev, next, threshold=thresholds[1])[0],
                        psi(prev, next, threshold=thresholds[2])[0],
                        wd(prev, next, threshold=thresholds[3])[0]]

                votes = [js(prev, next, threshold=thresholds[0])[1],
                        kl_div(prev, next, threshold=thresholds[1])[1],
                        psi(prev, next, threshold=thresholds[2])[1],
                        wd(prev, next, threshold=thresholds[3])[1]]
                stop = np.mean(scores) <= 0.15 and sum(votes) < 2   
                
                gc.collect()
                count += 1
                if (count > 1000):
                    raise Exception('Cannot generate data')
        else:
            next = gen_nextday(prev, False, dist)

            count = 0
            thresholds = np.random.normal(0.1, 0.001, 4)
            thresholds[thresholds < 0.07] = 0.07
            thresholds[thresholds > 0.3] = 0.3

            scores = [js(prev, next, threshold=thresholds[0])[0],
                    kl_div(prev, next, threshold=thresholds[1])[0],
                    psi(prev, next, threshold=thresholds[2])[0],
                    wd(prev, next, threshold=thresholds[3])[0]]

            votes = [js(prev, next, threshold=thresholds[0])[1],
                    kl_div(prev, next, threshold=thresholds[1])[1],
                    psi(prev, next, threshold=thresholds[2])[1],
                    wd(prev, next, threshold=thresholds[3])[1]]

            stop = 0.1 <= np.mean(scores) <= 0.3 and sum(votes) >= 2
            while not stop:
                next = gen_nextday(prev, False, dist)

                scores = [js(prev, next, threshold=thresholds[0])[0],
                        kl_div(prev, next, threshold=thresholds[1])[0],
                        psi(prev, next, threshold=thresholds[2])[0],
                        wd(prev, next, threshold=thresholds[3])[0]]

                votes = [js(prev, next, threshold=thresholds[0])[1],
                        kl_div(prev, next, threshold=thresholds[1])[1],
                        psi(prev, next, threshold=thresholds[2])[1],
                        wd(prev, next, threshold=thresholds[3])[1]]
                
                stop = 0.1 <= np.mean(scores) <= 0.3 and sum(votes) >= 2
                
                gc.collect()
                count += 1
                if (count > 1000):
                    raise Exception('Cannot generate data')
        # print("Final:", label, scores, np.mean(scores), votes)

        prev = next
        if (mode == 'histogram'):
            data = np.append(data, [hist(next) + [label]], axis=0)
        else:
            data = np.append(data, [next + [label]], axis=0)
        gc.collect()
    return data


def generate_data(num_days, num_samples, dist, mode='histogram', normal_ratio=0.9, visualize=False):
    '''
        num_days: number,
        num_sample: number,
        dist: 'normal' or 'logistic' or 'uniform' or 'mix',
    '''
    done = False
    data = np.array([])
    while not done:
        try:
            data = generate(num_days, num_samples, dist, mode, normal_ratio)
            done = True
        except:
            print('Error, retrying...')
            pass

    if visualize:
        plt.figure()
        for row in data[0:6, :]:
            if (mode == 'histogram'):
                y = gaussian_filter1d(row[:-1], sigma=1)
                sns.lineplot(x=range(300, 850), y=y, color='blue' if row[-1]
                             == 0 else 'red', linewidth=2)
            else:
                sns.kdeplot(row[:-1], color='blue' if row[-1]
                            == 0 else 'red', multiple='stack')
        plt.show()
        print(data[:5, :])

    return data