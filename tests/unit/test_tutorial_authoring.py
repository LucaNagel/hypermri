"""Execute the maintained contributor-facing scientific snippet."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_tutorial_authoring_snippet_executes_with_its_documented_units():
    """Execute the actual RST example so its API names cannot drift silently."""
    import textwrap

    import numpy as np

    path = PROJECT_ROOT / "docs/source/development/tutorial_authoring.rst"
    source = path.read_text(encoding="utf-8").split(".. code-block:: python\n", 1)[1]
    block = []
    for line in source.splitlines():
        if line and not line.startswith(" "):
            break
        block.append(line)
    scope = {}
    exec(compile(textwrap.dedent("\n".join(block)), path.name, "exec"), scope)
    assert scope["fid"].shape == (32,)
    np.testing.assert_allclose(scope["fid"][0], 1.0)
    np.testing.assert_allclose(scope["time_s"][-1], 0.0155)
