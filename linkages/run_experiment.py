import sys

import models
from sampler import Sampler


if __name__ == "__main__":
    filename = sys.argv[1]
    with open(filename, "r") as fd:
        sampler = Sampler(fd)

    sampler.summarize()

    model_ftm = models.fit_ftm(sampler)
    model_logit = models.fit_logit(sampler)
    model_xgb = models.fit_xgboost(sampler)

