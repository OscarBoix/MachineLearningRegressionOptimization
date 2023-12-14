import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection


def input_data():
    # Load the data from the file and split it into features and targets
    df = pd.read_csv("energy_performance.csv")

    data = np.array(df[["Relative compactness",
                        "Surface area",
                        "Wall area",
                        "Roof area",
                        "Overall height",
                        "Orientation",
                        "Glazing area",
                        "Glazing area distribution"]])

    target = np.array(df[["Heating load",
                          "Cooling load"]])

    # Determine and output the minimum and maximum heating and
    # cooling loads of buildings present in the dataset
    min_heating_load = min(target[0])
    max_heating_load = max(target[0])
    min_cooling_load = min(target[1])
    max_cooling_load = max(target[1])

    print(min_heating_load, max_heating_load, min_cooling_load, max_cooling_load)
    return data, target


def parameter_vector_size(degree):
    # The polynomial parameter vector size will be defined by the
    # amount of combinations that can be done with the degree.
    #
    # In this loop, we try every combination of variables in the
    # polynomial that does not exceed the degree.
    params_count = 0
    for i in range(degree + 1):
        for j in range(degree + 1):
            for k in range(degree + 1):
                for l in range(degree + 1):
                    for m in range(degree + 1):
                        for n in range(degree + 1):
                            for o in range(degree + 1):
                                for p in range(degree + 1):
                                    if i + j + k + l + m + n + o + p <= degree:
                                        params_count += 1
    return params_count


def calculate_model_function(degree, data, param_vector):
    # As in the parameter_vector_size, here we have a loop that iterates
    # through the polynomial possible combinations of feature vectors but only
    # selects those which do not exceed the degree, but here we calculate
    # the polynomial using the feature vectors and the parameter vector given.

    result = np.zeros(data.shape[0])
    c = 0
    for i in range(degree + 1):
        for j in range(degree + 1):
            for k in range(degree + 1):
                for l in range(degree + 1):
                    for m in range(degree + 1):
                        for n in range(degree + 1):
                            for o in range(degree + 1):
                                for p in range(degree + 1):
                                    if i + j + k + l + m + n + o + p <= degree:
                                        result += param_vector[c] * (data[:, 0] ** i) * (data[:, 1] ** j) * (
                                                data[:, 2] ** k) * (data[:, 3] ** l) * (data[:, 4] ** m) * (
                                                          data[:, 5] ** n) * (data[:, 6] ** o) * (data[:, 7] ** p)
                                        c += 1
    return result


def linearize(degree, data, p0):
    # We first call calculate_normal_function to create an initial instance of the value,
    # and then we add an epsilon difference to each parameter and call the function again
    # with the difference to determine how that parameter affected the function result,
    # in other words, we are calculating the partial derivatives or rate of change
    # of each feature vector versus each parameter.

    f0 = calculate_model_function(degree, data, p0)
    jac = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function(degree, data, p0)
        p0[i] -= epsilon
        di = (fi - f0) / epsilon
        jac[:, i] = di
    return f0, jac


def calculate_update(target, f0, jac):
    regularization_term = 1e-2

    # Calculate the normal equation matrix. It is regularised using a matrix
    # which has the regularization term value in the main diagonal.
    normal = jac.T @ jac + regularization_term * np.eye(jac.shape[1])

    # The residual here is calculated with the difference between the real
    # values and the values calculated.
    residual = target - f0
    gradient = jac.T @ residual
    dp = np.linalg.solve(normal, gradient)
    return dp


def regression(degree, data, target):
    max_iter = 10

    # p0 is the parameter vector, initially with 0s
    p0 = np.zeros(parameter_vector_size(degree))
    for i in range(max_iter):
        f0, jac = linearize(degree, data, p0)
        # We calculate the update that the parameter vector needs calling
        # the calculate update function and add it to p0, which is the
        # parameter vector
        update_vector = calculate_update(target, f0, jac)
        p0 += update_vector
    return p0


def k_fold_cross_validation(data, target, degree):
    accuracy_score = []

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

    # K-fold cross-validation splits into training and evaluation subsets
    for train_index, test_index in kf.split(data, target):
        train_data = data[train_index]
        train_target = target[train_index]
        test_data = data[test_index]
        test_target = target[test_index]

        # Get the params
        param = regression(degree, train_data, train_target)
        # Predict labels
        predicted_targets = calculate_model_function(degree, test_data, param)

        difference = abs(test_target - predicted_targets)
        accuracy_score.append(numpy.average(difference))

    avg_accuracy = np.average(accuracy_score)

    print(avg_accuracy)
    return avg_accuracy


def plot(target, predicted_target, title):
    plt.scatter(target, predicted_target)
    plt.xlabel("Target value")
    plt.ylabel("Predicted value")
    plt.title(title)
    plt.show()


def main():
    data, target = input_data()
    quality_heat = []
    quality_cool = []

    for i in range(3):
        # Heating load
        quality_heat.append(k_fold_cross_validation(data, target[:, 0], i))
        print(quality_heat)

        # Cooling load
        quality_cool.append(k_fold_cross_validation(data, target[:, 1], i))
        print(quality_cool)

    # Decide which degree is the most accurate, the degree with less difference
    degree_heat = quality_heat.index(min(quality_heat))
    degree_cool = quality_cool.index(min(quality_cool))

    params_heat = regression(degree_heat, data, target[:, 0])
    params_cool = regression(degree_cool, data, target[:, 1])

    predicted_targets_heat = calculate_model_function(degree_heat, data, params_heat)
    predicted_targets_cool = calculate_model_function(degree_cool, data, params_cool)

    # Plot the estimated loads against the true loads
    # for both the heating and the cooling case

    plot(target[:, 0], predicted_targets_heat, "Heating loads regression")
    plot(target[:, 1], predicted_targets_cool, "Cooling loads regression")


main()
