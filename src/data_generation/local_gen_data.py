import os
from multiprocessing import Pool

ratios = [0.7, 0.8, 0.9]

output_folder = '/work'


def gen_test():
    dists = ['normal', 'logistic', 'uniform', 'exponential', 'gamma']
    # nums_sam = [10000, 100000, 1000000, 5000000, 10000000]
    nums_sam = [10000, 100000, 1000000]
    nums_day = [183, 365, 365*3]
    # dists = ['normal']
    # nums_sam = [10000000]
    # nums_day = [365*3]
    iter_bundle = [(dist, num_samples, num_days, i, r)
                   for dist in dists
                   for num_samples in nums_sam
                   for num_days in nums_day
                   for r in ratios
                   for i in range(1, 4)
                   ]

    slice_iter_bundle = iter_bundle[:]
    print("Going to generate", len(slice_iter_bundle), "distributions")
    commands = []
    for dist, num_samples, num_days, index, r in slice_iter_bundle:
        os.makedirs(f"{output_folder}/test-{dist}", exist_ok=True)
        commands.append(
            f"python3 data_generation/gen_data.py --days {num_days} --samples {num_samples} --dist {dist} --ratio {r} --output {output_folder}/test-{dist}/test_{index}_{dist}_{num_days}_days_{num_samples}_samples_{int(r*100)}.npy")

    with Pool(4) as pool:
        pool.map(os.system, commands)


def gen_train():
    # nums_sam = [1000, 3000, 10000, 30000, 100000,
    #             300000, 1000000, 3000000, 10000000]
    nums_day = [30, 60, 120, 365, 365*2, 365*3]
    dists = ['normal', 'logistic', 'uniform', 'exponential', 'gamma']
    # dists = ['logistic']
    # nums_sam = [1000000]
    nums_sam = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    # nums_day = [365*3]
    iter_bundle = [(dist, num_samples, num_days, r)
                   for dist in dists
                   for num_samples in nums_sam
                   for num_days in nums_day
                   for r in ratios
                   ]

    slice_iter_bundle = iter_bundle[:]
    print("Going to generate", len(slice_iter_bundle), "distributions")

    commands = []
    for dist, num_samples, num_days, r in slice_iter_bundle:
        os.makedirs(f"{output_folder}/{dist}", exist_ok=True)
        commands.append(
            f"python3 data_generation/gen_data.py --days {num_days} --samples {num_samples} --dist {dist} --ratio {r} --output {output_folder}/{dist}/{dist}_{num_days}_days_{num_samples}_samples_{int(r*100)}.npy")

    with Pool(4) as pool:
        pool.map(os.system, commands)


gen_train()
