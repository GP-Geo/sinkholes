"""Validation is positives-only, and stays that way.

Negative patches reach the **train** split only. ``--add_val_negatives`` put a
fixed 1:1 negative set into validation as well; it was used by the ``valneg``
reruns and all 14 ``clean22`` runs, and it was retired on 2026-08-20 because it
inflates ``val/dice`` without making it discriminate — an empty prediction on an
empty mask scores dice 1.0 (``losses.py:21``), so at 1:1 the mean is roughly
``(1 + dice_on_positives) / 2``.

The flag itself is deliberately still there: it is part of the strict ``dataset``
resume fingerprint, so deleting it would strand every run trained with it. These
tests pin both halves of that arrangement — nothing can *submit* it any more, and
a run that already used it can still resume.
"""

import argparse
import logging
import re

import pytest

from conftest import REPO_ROOT

SUBMIT_ALL = REPO_ROOT / "scripts" / "submit_all.sh"
TRAIN_TEMPLATES = [
    REPO_ROOT / "scripts" / "train" / name
    for name in ("train_tattn.sh", "train_convlstm.sh", "train_control.sh")
]

#: A job line in submit_all.sh, i.e. one that hands a template to `job`.
JOB_LINE = re.compile(r"^\s*scripts/train/\S+\.sh\s")


def _live_lines(path):
    """The file's executable lines: comments and blanks dropped."""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            yield line


def test_no_submitted_job_turns_validation_negatives_on():
    """Every training job in submit_all.sh trains against a positives-only val set."""
    offenders = [line for line in _live_lines(SUBMIT_ALL)
                 if JOB_LINE.match(line) and "VAL_NEGS=yes" in line]
    assert offenders == [], (
        "these submit_all.sh jobs would train with negatives in the validation set, "
        "which inflates val/dice to about (1 + dice_on_positives)/2:\n  "
        + "\n  ".join(offenders)
    )


def test_submit_all_defines_no_val_negatives_variable():
    """The $VALNEG shorthand is gone, so it cannot be pasted into a new job."""
    assignments = [line for line in _live_lines(SUBMIT_ALL)
                   if re.match(r"^VALNEG\s*=", line)]
    assert assignments == [], f"VALNEG was reinstated in submit_all.sh: {assignments}"


@pytest.mark.parametrize("template", TRAIN_TEMPLATES, ids=lambda p: p.name)
def test_the_train_templates_default_to_positives_only(template):
    """VAL_NEGS survives for resumes, but no template may default it to 'yes'."""
    defaults = [line for line in _live_lines(template)
                if line.startswith("VAL_NEGS=")]
    assert defaults == ['VAL_NEGS="${VAL_NEGS:-no}"                # yes | no   '
                        'DEPRECATED: resume only'], (
        f"{template.name} changed how VAL_NEGS defaults: {defaults}"
    )


def test_a_run_trained_with_validation_negatives_can_still_resume():
    """The flag still parses, so the strict `dataset` fingerprint still matches.

    This is the one thing keeping ``--add_val_negatives`` in the codebase.
    ``dataset`` is a STRICT resume key and a run trained with the flag carries
    ``valneg=1-3x1.0`` in it, so the day the CLI stops accepting the flag is the
    day those runs have to be retrained from scratch rather than resumed.
    """
    from sinkholes.training import resume as R
    from sinkholes.training.train import add_arguments

    parser = argparse.ArgumentParser()
    add_arguments(parser)

    parsed = parser.parse_args(["--add_val_negatives", "--seed", "42"])
    assert parsed.add_val_negatives is True
    assert "valneg=1-3x1.0" in R._dataset_signature(parsed)

    parsed.add_val_negatives = False
    assert "valneg" not in R._dataset_signature(parsed), (
        "a positives-only run must keep the exact dataset signature it had before "
        "this option existed, or every older checkpoint stops resuming"
    )


def test_using_the_flag_warns_that_it_is_deprecated(caplog):
    """A run that passes it says so in its own log, before any training happens."""
    from sinkholes.training.train import add_arguments, warn_if_val_negatives_deprecated

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    log = logging.getLogger("test_val_negatives")

    with caplog.at_level(logging.WARNING, logger=log.name):
        warn_if_val_negatives_deprecated(
            parser.parse_args(["--add_val_negatives", "--seed", "42"]), log)
    assert any("DEPRECATED" in r.message for r in caplog.records), (
        f"no deprecation warning was logged: {[r.message for r in caplog.records]}"
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=log.name):
        warn_if_val_negatives_deprecated(parser.parse_args(["--seed", "42"]), log)
    assert caplog.records == [], "a positives-only run must not warn about anything"
