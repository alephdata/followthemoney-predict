from pathlib import Path
from functools import singledispatchmethod
from collections import Counter
from itertools import combinations
from enum import Enum
import json

from tqdm.autonotebook import tqdm
from followthemoney import model, compare2
from followthemoney.proxy import EntityProxy
from followthemoney.exc import InvalidData

from utils import (
    max_or_none,
    create_user_weights_lookup,
    calculate_pair_weights,
    TARGETS,
)

import numpy as np


def _describe_list(name, data):
    print(f"Mean {name}:", np.mean(data))
    print(f"STD {name}:", np.std(data))
    print(f"Median {name}:", np.median(data))


class Judgement(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNSURE = "unsure"
    NO_JUDGEMENT = "no_judgement"

    def __add__(self, other):
        pair = {self, other}
        if pair == {Judgement.POSITIVE}:
            return Judgement.POSITIVE
        elif pair == {Judgement.POSITIVE, Judgement.NEGATIVE}:
            return Judgement.NEGATIVE
        return Judgement.UNSURE

    def to_dict(self):
        return str(self.value)


class ProfileCollection(dict):
    @classmethod
    def load_dir(cls, directory):
        profiles = cls()
        for profile_file in Path(directory).glob("*.json"):
            with open(profile_file) as fd:
                for line in fd:
                    decision = json.loads(line)
                    profiles.add_decision(decision, quiet=True)
        return profiles

    def add_decision(self, decision, quiet=False):
        try:
            profile_id = decision["entityset_id"]
        except KeyError:
            if not quiet:
                raise ValueError("Not a valid decision")
            return
        profile = self.get(profile_id)
        if not profile:
            profile = self[profile_id] = Profile(profile_id)
        profile.add_decision(decision)

    def iter_decisions(self, judgements=None):
        for profile in self.values():
            yield from profile.iter_decisions(judgements)

    def iter_proxies(self, judgements=None):
        for profile in self.values():
            yield from profile.iter_proxies(judgements)

    def iter_pairs(self, judgements=None):
        for profile in self.values():
            yield from profile.iter_pairs(judgements)

    def describe(self):
        print("Number of profiles:", len(self))
        profile_lengths = [len(p) for p in self.values()]
        _describe_list("profile size", profile_lengths)
        print(
            "Unique Entities:",
            len({p.id for profile in self.values() for p in profile.proxies}),
        )
        print(
            "Judgement Counts:",
            Counter(judgement for _, judgement in self.iter_proxies()).most_common(),
        )
        proxy_n_properties = [
            len(p.properties) for profile in self.values() for p in profile.proxies
        ]
        _describe_list("entity num properties", proxy_n_properties)

    def to_pairs_dict(self, targets=TARGETS, judgements=None):
        pairs_scores = []
        user_weights = create_user_weights_lookup(self)
        N = len(targets)
        for profile in tqdm(self.values()):
            pairs = profile.iter_pairs(judgements)
            for (e1, e2), judgement in pairs:
                weights = calculate_pair_weights(e1, e2, profile, user_weights)
                scores = compare2.compare_scores(model, e1, e2)
                scores_str = {str(k): s for k, s in scores.items()}
                data = {}
                data.update(scores_str)
                data.update({f"has_{f}": True for f in scores_str.keys()})
                data["ftm_score"] = compare2._compare(scores)
                data["pct_full"] = sum(s is not None for s in scores.values()) / N
                data["pct_partial"] = sum(s is None for s in scores.values()) / N
                data["pct_empty"] = len(scores) / N
                data["left_id"] = e1.id
                data["right_id"] = e2.id
                data["user_weight"] = weights.user_weight
                data["pair_weight"] = weights.pair_weight
                data["judgement"] = judgement.value
                pairs_scores.append(data)
        return pairs_scores


class Profile:
    def __init__(self, pid):
        self.pid = pid
        self.proxies = {}
        self.decisions = {}

    def add_decision(self, decision):
        assert decision["entityset_id"] == self.pid
        judgement = decision["judgement"] = Judgement(decision["judgement"])
        self.decisions[decision["id"]] = decision
        try:
            data = model.get_proxy(decision.pop("entity"))
            self.proxies[model.get_proxy(data)] = judgement
        except (TypeError, InvalidData, KeyError):
            pass
        if decision["compared_to_entity_id"]:
            try:
                data = decision.pop("compared_to_entity")
                self.proxies.setdefault(model.get_proxy(data), Judgement.POSITIVE)
            except (TypeError, InvalidData, KeyError):
                pass

    def iter_decisions(self, judgements=None):
        if judgements is not None:
            for decision in self.decisions.values():
                if decision.get("judgement") in judgements:
                    yield decision
        else:
            yield from self.decisions.values()

    def iter_proxies(self, judgements=None):
        if judgements is not None:
            yield from ((p, j) for p, j in self.proxies.items() if j in judgements)
        else:
            yield from self.proxies.items()

    @singledispatchmethod
    def entity_decisions(self, arg):
        raise NotImplementedError("Argument must be an entity ID or a proxy", type(arg))

    @entity_decisions.register
    def _(self, eid: str):
        for decision in self.decisions.values():
            if decision["entity_id"] == eid or decision["compared_to_entity_id"] == eid:
                yield decision

    @entity_decisions.register
    def _(self, proxy: EntityProxy):
        yield from self.entity_decisions(proxy.id)

    def describe(self):
        print("Number of proxies:", len(self.proxies))
        print("Number of decisions:", len(self.decisions))
        print(
            "Judgement Counts:",
            Counter(d["judgement"] for d in self.decisions.values()).most_common(),
        )

    def __len__(self):
        return len(self.proxies)

    def __repr__(self):
        return f"<Profile({len(self)}) {self.pid}>"

    def iter_pairs(self, judgements=None):
        for (e1, j1), (e2, j2) in combinations(self.iter_proxies(), 2):
            if j1 is None or j2 is None:
                judgement = Judgement.UNSURE
            else:
                judgement = j1 + j2
            if judgements and judgement not in judgements:
                continue
            yield (e1, e2), judgement
