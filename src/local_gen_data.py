from data_generation.gen_data import generate_data
import os
import numpy as np

dists = ['normal', 'logistic', 'uniform', 'exponential', 'gamma']
# nums_sam = [10000, 100000, 1000000, 5000000, 10000000]
# nums_day = [183, 365, 365*3]
# dists = ['normal']
nums_sam = [10000000]
nums_day = [365*3]
iter_bundle = [(num_samples, num_days, i)
               for num_samples in nums_sam
               for num_days in nums_day
               for i in range(1, 4)
               ]


for dist in dists:
    os.makedirs(f"../../test-{dist}", exist_ok=True)
    for num_samples, num_days, i in iter_bundle[:]:
        data = generate_data(num_days, num_samples, dist, visualize=False)
        with open(f"../../test-{dist}/test_{i}_{dist}_{num_days}_days_{num_samples}_samples.npy", "wb") as f:
            np.save(f, data)
        del data
