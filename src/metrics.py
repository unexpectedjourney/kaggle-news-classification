from sklearn.metrics import f1_score


def f1_macro(actual, predicted):
    return f1_score(actual, predicted, average="macro")
