"""The tracked ``.gitignore`` must not name local study folders.

Study folders are named after the acquisition date and the operator's
initials.  The blanket ``examples`` rule already ignores them, so no entry
may spell such a name.
"""

import re
from pathlib import Path

import pytest

GITIGNORE = Path(__file__).resolve().parents[2] / ".gitignore"

# Study folder names start with a dotted date followed by " - ".
DATED_STUDY_FOLDER = re.compile(r"\d{4}\.\d{2}\.\d{2} - ")


@pytest.mark.skipif(not GITIGNORE.is_file(), reason="not a source checkout")
def test_gitignore_names_no_dated_study_folder():
    offending = [
        number
        for number, line in enumerate(GITIGNORE.read_text().splitlines(), start=1)
        if DATED_STUDY_FOLDER.search(line)
    ]

    assert offending == [], f".gitignore lines {offending} name a study folder"
