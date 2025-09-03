from itertools import product, combinations
from copy import copy
import sys

value = [10, 15, 7]
weight = [3, 4, 6]
capacity = 15


def obj(xlist, lam):
    V = -sum([v * x for v, x in zip(value, xlist)])
    P = 0
    valid = True

    W = sum([w * x for w, x in zip(weight, xlist)])
    # W <= capacity
    # capacity - W >= 0
    # capacity - W = slack
    # capacity - W = s0 + 2*s1 + 4*s2 + 8*s3
    # capacity - W - (s0 + 2*s1 + 4*s2 + 8*s3) == 0
    # (capacity - W - (s0 + 2*s1 + 4*s2 + 8*s3))**2
    S = sum([x * 2**i for i, x in enumerate(xlist[len(value) :])])
    P += (capacity - W - S) ** 2
    valid = capacity >= W
    return V + lam * P, valid


def knapsack(lam):
    min_obj = 10000000
    slack_vars = 4
    for xlist in product(range(2), repeat=len(value) + slack_vars):
        x_obj, valid = obj(xlist, lam)
        if x_obj < min_obj:
            min_obj = x_obj
            min_xlist = copy(xlist)
            min_valid = valid

    print(min_obj, min_xlist, min_valid)


usage = "knapsack.py bag_capacity"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    lam = float(sys.argv[1])
    knapsack(lam)
