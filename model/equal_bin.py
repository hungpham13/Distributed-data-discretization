from psi import calculate_psi
from psi import get_breakpoint
from data.gen_data import generate_data
from sklearn.metrics import confusion_matrix, classification_report


def run(num_tries):
    for _ in range(num_tries):
        train, test = generate_data(num_train=500,
                                    num_test=200,
                                    true_ratio=0.9,
                                    visualize=False,
                                    num_sample=10000,
                                    bin_num=5,
                                    choose_config=False
                                    )
        predict = []
        label = []
        data_distribution = train.drop(columns=['Label'])
        breakpoints = get_breakpoint(data_distribution.to_numpy().reshape(-1), buckettype='bins', buckets=10)

        for i in range(len(test)-1):
            psi = calculate_psi(expected=test.loc[i], actual=test.loc[i+1], breakpoints=breakpoints)
            label.append(test['Label'].loc[i] | test['Label'].loc[i+1])
            if psi > 0.1:
                predict.append(1)
            else:
                predict.append(0)

        print(confusion_matrix(label, predict))
        print(classification_report(label, predict))
