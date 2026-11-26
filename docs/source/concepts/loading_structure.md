# Inside a loaded scan

`Study("path/to/study")` returns a table of scans, and `study[8]` returns one
of them. This page shows what that object is made of: which part reads the
files, which part knows the sequence, and where the images end up.

## Three layers

```{mermaid}
flowchart TB
    subgraph scan["study[8]: a FLASH object"]
        direction TB
        seq["<b>Sequence class</b> (FLASH, CSI, NSPECT, ...)<br/>knows the sequence: which reconstructions,<br/>sequence-specific plots and fits"]
        exp["<b>MRIExperiment</b> (base of every sequence class)<br/>reconstruct(), plot(), masks,<br/>stored reconstructions"]
        h["<b>Vendor handler</b> (BrukerHandler, SiemensHandler, ...)<br/>reads the files: method, acqp, visu_pars,<br/>fid, 2dseq"]
        seq -- "is a" --> exp
        exp -- "scan.handler" --> h
    end
    files[("scan directory<br/>method, acqp, fid,<br/>pdata/1/visu_pars, 2dseq")]
    h -- "reads" --> files
```

The **vendor handler** is the only part that knows file formats. For a Bruker
scan it parses the parameter files when the scan is loaded and reads the large
`fid` and `2dseq` files only when they are needed.

**MRIExperiment** is vendor independent. It owns the handler and the
reconstructions, and provides what every scan can do: reconstruct, plot and
handle masks. A scan whose method HyperMRI does not know is a plain
`MRIExperiment`.

The **sequence class** adds what depends on the sequence: which
reconstructions to run by default, and sequence-specific tools such as
spectral fitting for CSI. Sequence classes derive from `MRIExperiment`, so a
`FLASH` object *is* an `MRIExperiment`.

```{mermaid}
classDiagram
    direction LR
    class MRIExperiment {
        handler
        params
        reconstructions
        reconstruct(reco_type)
        plot()
    }
    class BaseSequence {
        RECONSTRUCTION_POLICY
        run_default_reconstructions()
    }
    class BaseAnatomy
    class BaseSpectroscopy
    class VendorHandler {
        get_parameters()
        load_data()
    }
    MRIExperiment <|-- BaseSequence
    BaseSequence <|-- BaseAnatomy
    BaseSequence <|-- BaseSpectroscopy
    BaseAnatomy <|-- FLASH
    BaseAnatomy <|-- RARE
    BaseSpectroscopy <|-- CSI
    BaseSpectroscopy <|-- NSPECT
    VendorHandler <|-- BrukerHandler
    VendorHandler <|-- SiemensHandler
    MRIExperiment o-- VendorHandler : handler
```

## What happens when a scan is loaded

`Study` calls {func}`hypermri.load_scan` once for every scan directory.
`load_scan` is also the function to use for a single scan.

```{mermaid}
sequenceDiagram
    participant U as Study / user
    participant L as load_scan
    participant V as vendor registry
    participant H as BrukerHandler
    participant M as sequence mapping
    participant S as FLASH object
    participant R as reconstruction strategy
    U->>L: load_scan(path, mode="minimal")
    L->>V: which vendor recognises this directory?
    V->>H: construct the handler once
    H-->>L: parameters read (method, acqp, visu_pars)
    L->>M: method "<Bruker:FLASH>" → which class?
    M-->>L: FLASH (category anatomical)
    L->>S: FLASH(handler)
    L->>S: run_default_reconstructions("minimal")
    S->>R: reconstruct("seq2d")
    R-->>S: xarray.DataArray with x/y/z coordinates
    S-->>U: scan object
```

1. **Vendor detection.** The vendor registry asks each vendor whether it
   recognises the directory and constructs exactly one handler.
2. **Sequence lookup.** The method name from the parameters, for example
   `<Bruker:FLASH>` or `<User:my_flash_variant>`, is looked up in the
   packaged sequence mapping. It names the class and its category
   (anatomical or spectroscopy). An unknown method gives a plain
   `MRIExperiment` and a warning.
3. **Construction.** The sequence class is built from the handler. Nothing is
   reconstructed yet, and no viewer is created.
4. **Default reconstructions.** The loading mode decides what runs now (see
   below). Each reconstruction picks its strategy from a table keyed by
   vendor, sequence and kind (`"seq2d"` or `"raw"`).

## Loading modes

The `mode` of `load_scan` (`load_data` of `Study`) sets how much work is done
up front:

| mode | anatomy (e.g. FLASH) | spectroscopy (e.g. NSPECT) | CSI |
|---|---|---|---|
| `"lazy"` | parameters only | parameters only | parameters only |
| `"minimal"` (Study default) | scanner image (`seq2d`) | raw reconstruction | scanner image |
| `"full"` | scanner image and raw reconstruction | raw reconstruction | scanner image and raw reconstruction |

Some sequences set their own table; STEAM, for example, also loads the
scanner image in `"full"` mode. Anything not run up front can be run later with
`scan.reconstruct("raw")` or `scan.reconstruct("seq2d")`. A reconstruction
that already exists is not repeated.

## Where to find what

| what | where | example |
|---|---|---|
| ParaVision parameters | `scan.handler.method`, `scan.handler.acqp`, `scan.handler.visu` | `scan.handler.method["PVM_RepetitionTime"]` |
| standard parameters, same names for every vendor | `scan.params` or attributes | `scan.repetition_time_ms`, `scan.fov_mm` |
| scanner image (ParaVision `2dseq`) | `scan.scanner_data` | `scan.scanner_data.dims` |
| HyperMRI raw reconstruction | `scan.data` | `scan.reconstruct("raw")` |
| the image used by plots: `scan.data`, else `scan.scanner_data` | `scan.primary_data` | `scan.plot()` |
| all reconstructions by kind | `scan.reconstructions` | `scan.reconstructions["seq2d"]` |
| raw samples | `scan.handler.fid` | |

All images are `xarray.DataArray` objects with named dimensions, mm
coordinates and attributes; {doc}`data_model` describes them, and
{doc}`scanner_geometry` explains their x/y/z coordinates.

## See also

- {doc}`architecture` for the package layout behind these layers.
- {doc}`configuration` and {doc}`registry_overview` for the sequence mapping
  and how to add a custom method name.
- {doc}`reconstruction` for the difference between scanner image and raw
  reconstruction.
