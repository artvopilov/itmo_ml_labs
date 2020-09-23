import math


def manhattan_dist(a, b):
    return math.sqrt(sum([abs(a_i - b_i) for a_i, b_i in zip(a, b)]))


def euclidean_dist(a, b):
    return math.sqrt(sum([(a_i - b_i) ** 2 for a_i, b_i in zip(a, b)]))


def chebyshev_dist(a, b):
    return math.sqrt(max([abs(a_i - b_i) for a_i, b_i in zip(a, b)]))


def uniform_kernel_func(x):
    return 0.5 if abs(x) < 1 else 0


def triangular_kernel_func(x):
    return 1 - abs(x) if abs(x) < 1 else 0


def epanechnikov_kernel_func(x):
    return 0.75 * (1 - x ** 2) if abs(x) <= 1 else 0


def quartic_kernel_func(x):
    return 15 / 16 * (1 - x ** 2) ** 2 if abs(x) <= 1 else 0


def triweight_kernel_func(x):
    return 35 / 32 * (1 - x ** 2) ** 3 if abs(x) <= 1 else 0


def tricube_kernel_func(x):
    return 70 / 81 * (1 - abs(x) ** 3) ** 3 if abs(x) <= 1 else 0


def gaussian_kernel_func(x):
    return 1 / math.sqrt(2 * math.pi) * math.exp(-0.5 * x ** 2)


def cosine_kernel_func(x):
    return math.pi / 4 * math.cos(math.pi / 2 * x) if abs(x) <= 1 else 0


def logistic_kernel_func(x):
    return 1 / (math.exp(x) + 2 + math.exp(-x))


def sigmoid_kernel_func(x):
    return 2 / math.pi * 1 / (math.exp(x) + math.exp(-x))


def calculate_distance(dist, a, b):
    if dist == 'manhattan':
        return manhattan_dist(a, b)
    elif dist == 'euclidean':
        return euclidean_dist(a, b)
    else:
        return chebyshev_dist(a, b)


def calculate_weight(kern, x):
    if kern == 'uniform':
        return uniform_kernel_func(x)
    elif kern == 'triangular':
        return triangular_kernel_func(x)
    elif kern == 'epanechnikov':
        return epanechnikov_kernel_func(x)
    elif kern == 'quartic':
        return quartic_kernel_func(x)
    elif kern == 'triweight':
        return triweight_kernel_func(x)
    elif kern == 'tricube':
        return tricube_kernel_func(x)
    elif kern == 'gaussian':
        return gaussian_kernel_func(x)
    elif kern == 'cosine':
        return cosine_kernel_func(x)
    elif kern == 'logistic':
        return logistic_kernel_func(x)
    else:
        return sigmoid_kernel_func(x)


if __name__ == '__main__':
    n, m = map(lambda s: int(s), input().split())

    train_x, train_y = [], []
    for i in range(n):
        sample = list(map(lambda s: int(s), input().split()))
        x, y = sample[:-1], sample[-1]

        train_x.append(x)
        train_y.append(y)

    predict_x = list(map(lambda s: int(s), input().split()))

    distance_name = input()
    kernel_name = input()
    window_type = input()
    h = int(input())

    distances = []
    for i, x in enumerate(train_x):
        distance = calculate_distance(distance_name, predict_x, x)
        distances.append(distance)

    # print(distances)

    if window_type == 'variable':
        h = sorted(distances)[h]

    weights = []
    for d in distances:
        kern_x = d / (10e-6 if h == 0 else h)
        weights.append(calculate_weight(kernel_name, kern_x))

    # print(weights)
    # print(train_y)

    weights_sum = sum(weights)
    predict_y = sum([w * y for w, y in zip(weights, train_y)]) / weights_sum if weights_sum > 0 else 10e-6
    print(predict_y)
