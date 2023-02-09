import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy, ks_2samp, wasserstein_distance


def get_binned_data(reference, current, n_bins=30):
    bins = np.histogram_bin_edges(list(reference) + list(current), bins=n_bins)

    reference_percents = np.histogram(reference, bins)[0] / len(reference)
    current_percents = np.histogram(current, bins)[0] / len(current)

    np.place(reference_percents, reference_percents == 0, 1e-6)
    np.place(current_percents, current_percents == 0, 1e-6)

    return reference_percents, current_percents


def js(reference_data, current_data, threshold=0.1, n_bins=20):
    reference_percents, current_percents = get_binned_data(reference_data, current_data, n_bins)
    jensenshannon_value = distance.jensenshannon(reference_percents, current_percents)

    return jensenshannon_value, jensenshannon_value >= threshold


def kl_div(reference_data, current_data, threshold=0.1, n_bins=20):
    reference_percents, current_percents = get_binned_data(reference_data, current_data, n_bins)
    kl_div_value = entropy(reference_percents, current_percents)

    return kl_div_value, kl_div_value >= threshold


def ks(reference_data, current_data, threshold=0.1):
    p_value = ks_2samp(reference_data, current_data)[1]

    return p_value, p_value <= threshold


def psi(reference_data, current_data, threshold=0.1, n_bins=20):
    reference_percents, current_percents = get_binned_data(reference_data, current_data, n_bins)

    def sub_psi(ref_perc, curr_perc):
        """Calculate the actual PSI value from comparing the values.
            Update the actual value to a very small number if equal to zero
        """
        value = (ref_perc - curr_perc) * np.log(ref_perc / curr_perc)
        return value
    
    psi_value = 0
    for i, _ in enumerate(reference_percents):
        psi_value += sub_psi(reference_percents[i], current_percents[i])

    return psi_value, psi_value >= threshold


def wd(reference_data, current_data, threshold=0.1):
    norm = max(np.std(reference_data), 0.001)
    wd_norm_value = wasserstein_distance(reference_data, current_data) / norm
    
    return wd_norm_value, wd_norm_value >= threshold