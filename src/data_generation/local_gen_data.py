from gen_data import generate_data
import os
import numpy as np

ratios = [0.7, 0.8, 0.9]


def gen_test():
    dists = ['normal', 'logistic', 'uniform', 'exponential', 'gamma']
    nums_sam = [10000, 100000, 1000000, 5000000, 10000000]
    nums_day = [183, 365, 365*3]
    # dists = ['normal']
    # nums_sam = [10000000]
    # nums_day = [365*3]
    iter_bundle = [(num_samples, num_days, i, r)
                   for num_samples in nums_sam
                   for num_days in nums_day
                   for r in ratios
                   for i in range(1, 4)
                   ]

    for dist in dists:
        os.makedirs(f"../../test-{dist}", exist_ok=True)
        for num_samples, num_days, i, r in iter_bundle:
            data = generate_data(num_days, num_samples, dist, visualize=False)
            with open(f"../../test-{dist}/test_{i}_{dist}_{num_days}_days_{num_samples}_samples_{int(r*100)}.npy", "wb") as f:
                np.save(f, data)
            del data


def gen_train():
    nums_sam = [1000, 3000, 10000, 30000, 100000,
                300000, 1000000, 3000000, 10000000]
    nums_day = [30, 60, 120, 365, 365*2, 365*3]
    dists = ['normal', 'logistic', 'uniform', 'exponential', 'gamma']
    # dists = ['logistic']
    # nums_sam = [1000000]
    # nums_day = [365*3]
    iter_bundle = [(num_samples, num_days, r)
                   for num_samples in nums_sam
                   for num_days in nums_day
                   for r in ratios
                   ]

    # slice_iter_bundle = iter_bundle[39:len(iter_bundle)-3]
    print("Going to generate", len(iter_bundle), "distributions")

    for dist in dists:
        os.makedirs(f"/work/{dist}", exist_ok=True)
        for num_samples, num_days, r in iter_bundle:
            data = generate_data(num_days, num_samples, dist, visualize=False)
            with open(f"/work/{dist}/{dist}_{num_days}_days_{num_samples}_samples_{int(r*100)}.npy", "wb") as f:
                np.save(f, data)
            del data


gen_train()
gen_test()
