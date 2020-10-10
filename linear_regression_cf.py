import random
import math


def calculate_smape(predictions, test_y):
    errors_sum = sum([abs(p - y) / (abs(p) + abs(y) + 10e-5) for p, y in zip(predictions, test_y)])
    return errors_sum / len(test_y) * 100


def calculate_mape(predictions, test_y):
    errors_sum = sum([abs(p - y) for p, y in zip(predictions, test_y)])
    return errors_sum / len(test_y)


def calculate_mse(predictions, test_y):
    return sum([(p - y) ** 2 for p, y in zip(predictions, test_y)]) / len(test_y)


def calculate_smape_score(y_test, predictions):
    smape = calculate_smape(y_test, predictions)
    return 100 * (0.02 - smape) / (0.02 - 0.01)


def predict(weights, x_sample):
    return sum([w * x for w, x in zip(weights, x_sample)])


def sign(x):
    return math.copysign(1, x)


def calculate_gradient_for_smape(weights, x_sample, y):
    p = predict(weights, x_sample)
    gradient = [
        100 * (x * sign(p - y) * (abs(p) + abs(y)) - sign(p) * x * abs(p - y)) / ((abs(p) + abs(y)) ** 2 + 10e-5)
        for x, w in zip(x_sample, weights)
    ]
    return gradient


def calculate_gradient_for_mape(weights, x_sample, y):
    p = predict(weights, x_sample)
    return [(p - y) / abs(p - y) * x for x, w in zip(x_sample, weights)]


def calculate_gradient_for_mse(weights, x_sample, y):
    prediction = predict(weights, x_sample)
    return [(prediction - y) * x + w for x, w in zip(x_sample, weights)]


def update_weights(weights, x_batch, y_batch, lr):
    weights_gradients_sum = [0] * len(weights)
    for i in range(len(x_batch)):
        cur_weights_gradient = calculate_gradient_for_mape(weights, x_batch[i], y_batch[i])
        weights_gradients_sum = [grad_sum + cur_grad
                                 for grad_sum, cur_grad in zip(weights_gradients_sum, cur_weights_gradient)]
    # if e_i % 1 == 0:
    #     print('Gradients: {}'.format(' '.join([str(w) for w in weights_gradients_sum])))
    weights = [weight - lr * weight_grad
               for weight, weight_grad in zip(weights, weights_gradients_sum)]
    return weights


def normalize_train_x(train_x):
    max_x_list = []
    min_x_list = []
    for sample_x in train_x:
        if len(max_x_list) == 0:
            max_x_list = sample_x
            min_x_list = sample_x
        max_x_list = [max(x_1, x_2) for x_1, x_2 in zip(max_x_list, sample_x)]
        min_x_list = [min(x_1, x_2) for x_1, x_2 in zip(min_x_list, sample_x)]

    train_x_normalized = []
    for sample_x in train_x:
        train_x_normalized.append([(x - min_x) / (max_x - min_x + 10e-5)
                                   for x, max_x, min_x in zip(sample_x, max_x_list, min_x_list)])

    return train_x_normalized, max_x_list, min_x_list


def add_bias_feature(train_x):
    train_x_with_bias_feature = []
    for i in range(len(train_x)):
        train_x_with_bias_feature.append(train_x[i] + [1.0])

    return train_x_with_bias_feature


if __name__ == '__main__':
    n, m = map(lambda s: int(s), input().split())

    train_x, train_y = [], []
    for i in range(n):
        sample = list(map(lambda s: float(s), input().split()))
        x, y = sample[:-1], sample[-1]

        train_x.append(x)
        train_y.append(y)

    train_x_normalized, max_x_list, min_x_list = normalize_train_x(train_x)
    train_x_normalized = add_bias_feature(train_x_normalized)
    # print(train_x_normalized)

    mean_abs_y = sum(list(map(lambda x: abs(x), train_y))) / len(train_y)

    # weights = [(random.random() * 2 - 1) * max(train_y) for _ in range(m)] + [0]
    # weights = [(random.random() * 2 - 1) * mean_abs_y * 0.01 for _ in range(m)] + [0]
    # weights = [(random.random() * 2 - 1) * mean_abs_y * 0.01 for _ in range(m + 1)]
    weights = [random.random() * mean_abs_y * 0.01 for _ in range(m + 1)]
    # weights = [(random.random() * 2 - 1) * max(train_y) for _ in range(m + 1)]
    # weights = [random.random() for _ in range(m + 1)]
    # weights = [0 for _ in range(m + 1)]

    epochs = 200
    batch_size = max(math.ceil(len(train_x) / 10000), 100)
    lr = mean_abs_y * 0.1
    # lr = 0.01

    for e_i in range(epochs):
        # if e_i % 1 == 0:
        # print('Wights: {}'.format(' '.join([str(w) for w in weights])))
        # predictions = [predict(weights, x_sample) for x_sample in train_x_normalized]
        # print('Predictions: {}'.format(' '.join([str(p) for p in predictions])))
        # print('Epoch: {}, smape error: {}'.format(e_i, calculate_smape(predictions, train_y)))
        # print('Epoch: {}, mape error: {}'.format(e_i, calculate_mape(predictions, train_y)))
        # print('Epoch: {}, mse error: {}'.format(e_i, calculate_mse(predictions, train_y)))

        for b_i in range(0, len(train_x), batch_size):
            weights = update_weights(
                weights,
                train_x_normalized[batch_size * b_i:batch_size * b_i + batch_size],
                train_y[batch_size * b_i:batch_size * b_i + batch_size],
                lr
            )

    # print('Wights: {}'.format(' '.join([str(w) for w in weights])))
    # predictions = [predict(weights, x_sample) for x_sample in train_x_normalized]
    # print('Predictions: {}'.format(' '.join([str(p) for p in predictions])))
    # print('Epoch: {}, smape error: {}'.format(e_i, calculate_smape(predictions, train_y)))
    # print('Epoch: {}, mape error: {}'.format(e_i, calculate_mape(predictions, train_y)))
    # print('Epoch: {}, mse error: {}'.format(e_i, calculate_mse(predictions, train_y)))

    transformed_weights = []
    w_bias = 0
    for i in range(len(weights) - 1):
        scale = max_x_list[i] - min_x_list[i]
        offset = min_x_list[i]

        transformed_weights.append(weights[i] / scale)
        w_bias += offset * weights[i] / scale
    transformed_weights.append(weights[-1] - w_bias)

    for w in transformed_weights:
        print(w)
