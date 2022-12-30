from model.psi import calculate_psi
from model.psi import get_breakpoint
from data.gen_data import generate_data
from sklearn.metrics import confusion_matrix, classification_report

import warnings


def predict(test, breakpoints):
    prediction = []
    label = []
    for i in range(len(test) - 1):
        psi = calculate_psi(expected=test.loc[i], actual=test.loc[i + 1],
                            breakpoints=breakpoints)
        label.append(test['Labels'].loc[i] | test['Labels'].loc[i + 1])
        if psi > 0.2:
            prediction.append(1)
        else:
            prediction.append(0)

    print(confusion_matrix(label, prediction))
    print(classification_report(label, prediction))


# def run(num_tries=1, num_train=500, num_test=200):
#     warnings.filterwarnings("ignore", category=DeprecationWarning)
#     for _ in range(num_tries):
#         train, test = generate_data(num_train=num_train,
#                                     num_test=num_test,
#                                     true_ratio=0.9,
#                                     visualize=True,
#                                     num_sample=10000,
#                                     bin_num=1,
#                                     choose_config=False
#                                     )
#         data_distribution = train.drop(columns=['Label'])
#         breakpoints = get_breakpoint(data_distribution.to_numpy().reshape(-1),
#                                      buckettype='bins', buckets=10)
#         predict(test, breakpoints)
