---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
language_info:
  name: python
  pygments_lexer: ipython3
---

# Loading data

HyperMRI loads a scan folder once, detects the vendor and the sequence, and
returns an object that knows the scan's parameters and reconstructions. This
tutorial shows the two entry points, a whole session and a single scan.

```{code-cell} ipython3
from pathlib import Path

study_path = Path("20200612_094625_lego_phantom_3_1_2")
```

```{code-cell} ipython3
:tags: [remove-cell]

# The documentation build reads its local copy of the phantom.
import os

study_path = Path(os.environ["HYPERMRI_DOCS_PHANTOM"])
```

## A whole session

{class}`~hypermri.Study` collects every scan below a folder. By default it
sorts them by acquisition time and runs each sequence's *minimal*
reconstruction, which for Bruker data reads the scanner image (`2dseq`).

```{code-cell} ipython3
from hypermri import Study

study = Study(study_path, verbose=False)
len(study)
```

The scan table is a {class}`pandas.DataFrame`; in Jupyter, `Study(path)`
without `verbose=False` shows it coloured by sequence.

```{code-cell} ipython3
table = study.dataframe
table[["Sequence Type", "Scan Name", "Measurement Time"]]
```

Scans are indexed by their scan (experiment) number, as in ParaVision:

```{code-cell} ipython3
flash = study[3]
type(flash).__name__, flash.path.name
```

Scans that could not be loaded are not dropped silently; they are listed with
the reason:

```{code-cell} ipython3
study.load_errors
```

## Loading modes

Reading data costs time and memory, so the amount is chosen explicitly. Both
{class}`~hypermri.Study` (`load_data=`) and {func}`~hypermri.load_scan`
(`mode=`) accept the same three modes:

| Mode | Reads | Use |
| --- | --- | --- |
| `"lazy"` | parameters only | browsing a session, building tables |
| `"minimal"` | parameters and the scanner image | looking at images (default of `Study`) |
| `"full"` | also raw data, all default reconstructions | reconstruction and fitting |

```{code-cell} ipython3
lazy_study = Study(study_path, verbose=False, load_data="lazy")
lazy_study[3].get_available_reconstructions()
```

A lazy scan still reconstructs on request; `reconstruct` reads what it needs
({doc}`reconstruction`).

## A single scan

{func}`~hypermri.load_scan` loads one scan folder. Its default mode is
`"lazy"`.

```{code-cell} ipython3
from hypermri import load_scan

csi = load_scan(study_path / "30", mode="minimal")
csi
```

## Parameters

Each scan exposes vendor-independent parameter names with units in the name.
{meth}`~hypermri.MRIExperiment.get_parameters` lists them and
{meth}`~hypermri.MRIExperiment.get` reads one:

```{code-cell} ipython3
names = flash.get_parameters()
len(names), sorted(names)[:8]
```

```{code-cell} ipython3
{
    "fov_mm": flash.get("fov_mm"),
    "matrix": flash.get("matrix"),
    "repetition_time_ms": flash.get("repetition_time_ms"),
    "echo_time_ms": flash.get("echo_time_ms"),
    "flipangle_deg": flash.get("flipangle_deg"),
}
```

The study table can show any of these names as a column:

```{code-cell} ipython3
study.add_column("repetition_time_ms")
study.dataframe[["Sequence Type", "repetition_time_ms"]].head()
```

## Next

- {doc}`reconstruction`: reconstruct the FLASH scan from raw k-space.
- {doc}`../concepts/data_model`: dimensions, coordinates and attributes of the
  loaded data.
