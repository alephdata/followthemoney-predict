import os
import pickle
from itertools import islice

import models
import util
from sampler import Sampler, create_collection_pseudopairs


if __name__ == "__main__":
    with open("./data/linkages/linkages-latest.json", "r") as fd:
        linkages = list(util.load_links(fd))

    N = len(linkages)
    collections_dir = "./data/collections"
    for collection_file in os.listdir(collections_dir):
        if not collection_file.endswith(".json"):
            continue
        try:
            with open(
                os.path.join(collections_dir, collection_file + ".cache.pkl"), "rb"
            ) as fd:
                print("Loading collection link cache for", collection_file)
                collection_links = pickle.load(fd)
        except FileNotFoundError:
            print("Calculating collection links for", collection_file)
            with open(os.path.join(collections_dir, collection_file)) as fd:
                collection = list(util.load_collection(fd))
            collection_links = list(create_collection_pseudopairs(collection))
            with open(
                os.path.join(collections_dir, collection_file + ".cache.pkl"), "wb+"
            ) as fd:
                print("Caching connections")
                pickle.dump(collection_links, fd)
        linkages.extend(collection_links)

    sampler = Sampler(linkages)
    sampler.summarize()

    model_ftm = models.fit_ftm(sampler)
    model_logit = models.fit_logit(sampler)
    model_xgb = models.fit_xgboost(sampler)

