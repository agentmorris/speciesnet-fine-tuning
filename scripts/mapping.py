#%% Header

"""
mapping.py

Load and apply an optional category mapping, used at training time to rename,
merge, or drop classes. Keeping this decision at training time (rather than in
the data CSV) lets the data CSV stay a literal record of your labels.

The mapping CSV has columns "input" and "output" (any extra columns, such as the
"count" column written by coco_to_mapping_file.py, are ignored). For each row:

  * an empty (or absent) "output" means "leave this category unchanged", which
    is identical to the category not appearing in the file at all;
  * the reserved "output" value "remove" drops the category entirely (its
    instances are excluded from training);
  * any other "output" renames the category; several inputs sharing one output
    are merged into a single class.

Mapping is a single pass (no chaining): if you map A -> B and B -> C, an A
becomes B, not C. A duplicate "input" is an error.
"""

#%% Imports and constants

import csv

# Reserved value in the "output" column meaning "drop this category".
REMOVE_TOKEN = "remove"


#%% Support functions

def load_mapping(path):
    """
    Read a category mapping CSV.

    Args:
        path (str): path to the mapping CSV; must have "input" and "output"
            columns (any other columns are ignored)

    Returns:
        dict: maps each input category name to its output category name, or to
        None for categories to drop (output "remove"). Rows with an empty output
        are omitted, since leaving a category unchanged is identical to not
        listing it at all

    Raises:
        ValueError: if the file is empty, is missing the "input" or "output"
            column, has an empty "input" value, or lists the same input twice
    """

    mapping = {}

    with open(path, newline="", encoding="utf-8-sig") as f:

        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("mapping file '%s' is empty" % path)
        norm = {name: (name or "").strip().lower() for name in reader.fieldnames}
        if "input" not in norm.values() or "output" not in norm.values():
            raise ValueError(
                "mapping file '%s' must have columns 'input' and 'output' "
                "(found: %s)" % (path, reader.fieldnames))
        col = {v: k for k, v in norm.items()}

        for line_num, row in enumerate(reader, start=2):  # line 1 is the header

            inp = (row.get(col["input"]) or "").strip()
            out = (row.get(col["output"]) or "").strip()
            if inp == "":
                raise ValueError(
                    "mapping file '%s', line %d: empty 'input' value" % (path, line_num))
            if inp in mapping:
                raise ValueError(
                    "mapping file '%s', line %d: input category '%s' is listed more "
                    "than once" % (path, line_num, inp))
            if out == "":
                continue  # leave this category unchanged
            mapping[inp] = None if out.lower() == REMOVE_TOKEN else out

        # ...for each line

    # ...with open(...)

    return mapping


def apply_mapping(category, mapping):
    """
    Apply a mapping to a single category name.

    Args:
        category (str): a category name from the data
        mapping (dict): a mapping as returned by load_mapping()

    Returns:
        str: the mapped category name, or the original [category] if it is not
        listed in [mapping]; None if the category should be dropped (it was mapped
        to "remove")
    """

    return mapping.get(category, category)


def mapping_warnings(mapping, known_categories):
    """
    Find likely mistakes in a mapping, returned as non-fatal warning strings.

    Flags mapping inputs that don't appear in the data (likely typos), and outputs
    that are themselves inputs (which would imply chaining, which is not done).

    Args:
        mapping (dict): a mapping as returned by load_mapping()
        known_categories (set): the category names actually present in the data

    Returns:
        list: a list of warning strings (str); empty if no problems are found
    """

    warnings = []
    for inp in mapping:
        if inp not in known_categories:
            warnings.append(
                "mapping input '%s' does not appear in the data; the rule has no "
                "effect." % inp)
    outputs = {v for v in mapping.values() if v is not None}
    for out in outputs:
        if out in mapping:
            warnings.append(
                "mapping output '%s' is also a mapping input; mapping is single-pass "
                "(not chained), so it is left as '%s'." % (out, out))
    return warnings
