def calculate_f_measures(c_to_tp, c_to_fn, c_to_fp, c_to_n):
    c_to_f1 = {}
    precision_sum = 0
    recall_sum = 0
    for c in c_to_tp.keys():
        tp = c_to_tp[c]
        fn = c_to_fn[c]
        fp = c_to_fp[c]
        n = c_to_n[c]

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        c_to_f1[c] = 2 * precision * recall / (precision + recall + 1e-6) * n

        precision_sum += precision * n
        recall_sum += tp

    macro_precision = precision_sum / sum(c_to_n.values())
    macro_recall = recall_sum / sum(c_to_n.values())

    micro_f1 = sum(c_to_f1.values()) / sum(c_to_n.values())
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

    return micro_f1, macro_f1


if __name__ == '__main__':
    k = int(input())
    matrix = []
    for _ in range(k):
        matrix.append(list(map(lambda x: int(x), input().split())))

    c_to_tp = {}
    c_to_fn = {}
    c_to_fp = {}
    c_to_n = {}
    for c in range(len(matrix)):
        c_to_fn[c] = 0
        c_to_n[c] = 0
        for i, p in enumerate(matrix[c]):
            if i == c:
                c_to_tp[c] = p
                c_to_n[c] += p
                continue

            c_to_fn[c] += p
            c_to_n[c] += p

            if i not in c_to_fp:
                c_to_fp[i] = 0
            c_to_fp[i] += p

    micro_f1, macro_f1 = calculate_f_measures(c_to_tp, c_to_fn, c_to_fp, c_to_n)

    print(micro_f1)
    print(macro_f1)
