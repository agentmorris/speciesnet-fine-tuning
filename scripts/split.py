#%% Header

"""
split.py

Assign cameras (locations) to a train or validation split. Every image from a
given camera goes entirely to one side, so the model is validated on cameras it
never saw during training.

The split tries to put about val_fraction of the instances into validation, and
also about val_fraction of *each category's* instances into validation. Those
goals usually can't all be met (a rare species may live at a single camera, so
it is either 0% or 100% in validation), so this is a best effort: a greedy search
moves cameras into validation one at a time, each time choosing the camera that
most reduces the total squared deviation from the targets, and stops when no move
helps. The achieved balance is returned in the report and should be recorded in
the run summary.

Splits are measured in instances (boxes), not images, because one image can
contribute several instances.
"""

#%% Imports and constants

import random
from collections import Counter, defaultdict


#%% Imports and constants

def make_split(instances, val_fraction=0.15, seed=0):
    """
    Assign each camera (location) to the train or validation split.

    Uses the greedy search described in the module header: it tries to put about
    [val_fraction] of the instances, and about [val_fraction] of each category's
    instances, into validation, while keeping every camera entirely on one side.

    Args:
        instances (list of Instance): the instances to split; each has a .location
            and a .category, and counts as one instance (box)
        val_fraction (float, optional): target fraction of instances to place in
            validation, both overall and per category (default 0.15)
        seed (int, optional): random seed, used only to break ties between equally
            good moves so the split is reproducible (default 0)

    Returns:
        tuple: a 2-tuple (split, report). split (dict) maps each location to
        "train" or "val". report (dict) describes the achieved balance, with keys
        val_fraction_requested, seed, n_locations, n_train_locations,
        n_val_locations, frac_val_locations, n_instances_total, n_instances_train,
        n_instances_val, frac_val_instances, per_category (category -> {"train",
        "val", "total", "val_fraction"}), classes_missing_train, and
        classes_missing_val
    """

    # Tally instances per (location, category), per location, and per category, so
    # the greedy search below can score how moving each camera would shift the split.
    loc_counts = defaultdict(Counter)   # location -> {category: n}
    loc_total = Counter()               # location -> n instances
    total = Counter()                   # category -> n instances
    for inst in instances:
        loc_counts[inst.location][inst.category] += 1
        loc_total[inst.location] += 1
        total[inst.category] += 1
    total_all = sum(total.values())
    categories = list(total)
    t = val_fraction

    # Squared deviation of the overall validation fraction (across all instances)
    # from the target [t], as a function of a candidate validation instance count
    # [vt]. This is the one "overall balance" term added to the per-category terms.
    def overall_term(vt):
        r = (vt / total_all) if total_all else 0.0
        return (r - t) ** 2

    # Greedy-search state: every camera starts in train, and we move cameras into
    # val one at a time below. val_counts/val_total track what is in val so far.
    val = set()
    val_counts = Counter()
    val_total = 0

    remaining = list(loc_counts.keys())
    random.Random(seed).shuffle(remaining)  # deterministic tie-breaking

    # Greedily move one camera into val at a time: at each step pick the camera
    # whose move most reduces the total squared deviation from the per-category and
    # overall targets, and stop once no remaining move improves the balance.
    while True:

        best_loc = None
        best_delta = -1e-12  # require a strictly improving (negative) move
        for loc in remaining:
            delta = 0.0
            for c, cnt in loc_counts[loc].items():
                before = (val_counts[c] / total[c] - t) ** 2
                after = ((val_counts[c] + cnt) / total[c] - t) ** 2
                delta += after - before
            delta += overall_term(val_total + loc_total[loc]) - overall_term(val_total)
            if delta < best_delta:
                best_delta = delta
                best_loc = loc
        if best_loc is None:
            break
        for c, cnt in loc_counts[best_loc].items():
            val_counts[c] += cnt
        val_total += loc_total[best_loc]
        val.add(best_loc)
        remaining.remove(best_loc)

    all_locs = list(loc_counts.keys())

    # Guards for degenerate cases
    if not val and all_locs:
        best = min(all_locs, key=lambda L: abs(loc_total[L] / total_all - t))
        val.add(best)
        val_total = loc_total[best]
        val_counts = Counter(loc_counts[best])

    train = [L for L in all_locs if L not in val]

    # If the greedy search put every camera in val (so there is nothing left to
    # train on), move the single largest camera back to the training side.
    if not train and all_locs:

        biggest = max(all_locs, key=lambda L: loc_total[L])
        val.discard(biggest)
        for c, cnt in loc_counts[biggest].items():
            val_counts[c] -= cnt
        val_total -= loc_total[biggest]
        train = [biggest]

    split = {L: ("val" if L in val else "train") for L in all_locs}

    per_cat = {}
    classes_missing_train = []
    classes_missing_val = []

    # Build the per-category balance report, flagging any category that ended up
    # entirely on one side (no training instances, or no validation instances).
    for c in sorted(categories):

        vc = val_counts[c]
        tc = total[c]
        trc = tc - vc
        per_cat[c] = {"train": trc, "val": vc, "total": tc,
                      "val_fraction": (vc / tc) if tc else 0.0}
        if trc == 0:
            classes_missing_train.append(c)
        if vc == 0:
            classes_missing_val.append(c)

    report = {
        "val_fraction_requested": t,
        "seed": seed,
        "n_locations": len(all_locs),
        "n_train_locations": len(train),
        "n_val_locations": len(val),
        "frac_val_locations": (len(val) / len(all_locs)) if all_locs else 0.0,
        "n_instances_total": total_all,
        "n_instances_train": total_all - val_total,
        "n_instances_val": val_total,
        "frac_val_instances": (val_total / total_all) if total_all else 0.0,
        "per_category": per_cat,
        "classes_missing_train": classes_missing_train,
        "classes_missing_val": classes_missing_val,
    }

    return split, report

# ...def make_split(...)
