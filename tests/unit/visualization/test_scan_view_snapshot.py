"""HTML snapshot of the browser viewer (ScanView.to_html).

The snapshot must carry the values the widget sent to the browser unchanged,
switch the viewer into snapshot mode without tools, keep every script block
intact, and leave the view itself unchanged. The browser behaviour of the
page (voxel choice, curve position, colour range without Python) was checked
by hand in Chrome; these tests cover the page contents.
"""

import base64
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("anywidget")
from plotly.offline import get_plotlyjs_version  # noqa: E402

# Pytest puts this test directory on sys.path (rootdir-less "prepend" import
# mode), so the sibling module imports by its basename even under python -I.
from test_scan_view import (  # noqa: E402
    _stacked_anatomy_scan,
    _stacked_csi,
    _view,
)
from test_scan_view import _state as _plot_state  # noqa: E402

from hypermri.visualization.scan_view import plot_scan  # noqa: E402

_STATE = re.compile(
    r'<script id="hypermri-state" type="application/json">(.*?)</script>', re.S
)


def _state(page):
    return json.loads(_STATE.search(page).group(1))


def test_snapshot_carries_the_block_and_selection_unchanged():
    view, _ = _view()
    view.name = "Synthetic </script> CSI"
    state = _state(view.to_html(plotly_js="cdn"))
    block = base64.b64decode(state["_block"]["base64"])
    assert block == bytes(view._block)
    np.testing.assert_array_equal(
        np.frombuffer(block, dtype=np.float32),
        np.frombuffer(bytes(view._block), dtype=np.float32),
    )
    assert state["selection"] == view.selection
    assert state["_view"]["shape"] == list(view._view["shape"])
    assert state["name"] == view.name
    assert "_esm" not in state and "layout" not in state


def test_snapshot_drops_tools_and_marks_the_layout():
    view, _ = _view()
    view.tool = "masks"
    state = _state(view.to_html(plotly_js="cdn"))
    assert state["_layout"]["snapshot"] is True
    assert state["_layout"]["tools"] == []
    assert state["tool"] == ""
    assert view.tool == "masks"
    assert "snapshot" not in view._layout


def test_snapshot_script_blocks_are_not_cut_short():
    view, _ = _view()
    view.name = "a </script> name"
    page = view.to_html(plotly_js="inline")
    assert page.count("<script") == page.count("</script>") == 3
    assert "<title>a &lt;/script&gt; name</title>" in page


def test_snapshot_plotly_source(tmp_path):
    view, _ = _view()
    inline = view.to_html(tmp_path / "inline.html")
    cdn = view.to_html(plotly_js="cdn")
    assert (tmp_path / "inline.html").read_text(encoding="utf-8") == inline
    version = get_plotlyjs_version()
    assert view._plotly_version == version
    assert f"plotly.js-dist-min@{version}/plotly.min.js" in cdn
    assert len(inline) - len(cdn) > 3_000_000


def test_snapshot_rejects_an_unknown_plotly_source():
    view, _ = _view()
    with pytest.raises(ValueError, match="plotly_js must be one of"):
        view.to_html(plotly_js="local")


def test_snapshot_saves_every_anatomy_slice_and_keeps_the_view():
    scan = SimpleNamespace(primary_data=_stacked_csi(), is_spectroscopic=True)
    # Invented anatomy whose pixel value also tells its slice: x + 100 y +
    # 1000 z (mm), so every anatomy slice is a different plane.
    anatomy = _stacked_anatomy_scan()
    anatomy.primary_data = anatomy.primary_data + 1000.0 * anatomy.primary_data.z
    view = plot_scan(scan, anatomy=anatomy, state=_plot_state())
    view.set_anatomy_slice(4)
    before = {
        "state": repr(view.state),
        "controls": view._controls,
        "block": bytes(view._block),
        "anatomy": bytes(view._anatomy),
        "selection": view.selection,
    }
    state = _state(view.to_html(plotly_js="cdn"))

    slices = state["_anatomy_slices"]
    assert sorted(slices) == ["0", "1", "2", "3", "4", "follow"]
    # Each choice carries what the notebook slider would show: the drawn
    # anatomy slice, the controls and, for this stacked map, the map slice
    # nearest to it (anatomy z = -4..4 mm, map z = 0, 4, 8 mm).
    for key, saved in slices.items():
        assert {"_anatomy", "_controls", "_view"} <= set(saved)
        # Following the map at z = 4 mm draws anatomy slice 4 as well.
        expected = 4 if key == "follow" else int(key)
        assert saved["_view"]["anatomy"]["slice_index"] == expected
    assert slices["4"]["_controls"]["indices"]["slice_idx"] == 1
    assert slices["follow"]["_controls"]["anatomy"] == {"slice": None}
    assert slices["4"]["_anatomy"] == state["_anatomy"]
    assert slices["0"]["_anatomy"] != state["_anatomy"]

    after = {
        "state": repr(view.state),
        "controls": view._controls,
        "block": bytes(view._block),
        "anatomy": bytes(view._anatomy),
        "selection": view.selection,
    }
    assert after == before


def test_snapshot_without_anatomy_saves_no_slices():
    view, _ = _view()
    assert "_anatomy_slices" not in _state(view.to_html(plotly_js="cdn"))
