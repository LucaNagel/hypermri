# Data model

Every image, spectrum and fit result in HyperMRI is a labelled
{class}`xarray.DataArray` or {class}`xarray.Dataset`. Axes are addressed by
name, never by position, and physical units travel with the data.

## Dimensions

Imaging reconstructions, both HyperMRI's own (`"raw"`) and the scanner's
(`"seq2d"`), share one dimension order, owned by
{data}`hypermri.reconstruction.layout.IMAGING_DIM_ORDER`:

| Dimension | Meaning |
| --- | --- |
| `read_idx`, `phase_idx`, `slice_idx` | Voxel indices along read, phase and slice direction |
| `waterfat_idx` | Water/fat images of a Dixon scan |
| `inversion_idx` | Inversion times |
| `echo_idx` | Echoes |
| `bvalue_idx` | Diffusion weightings |
| `channel_idx` | Receive coils |
| `acquisition_idx` | Samples of one readout: the FID or spectrum |
| `repetition_idx` | Repetitions, time frames of a dynamic scan |

Dimensions a scan does not have are present with size 1, so the same code
works for every imaging scan: `data.isel(slice_idx=2, echo_idx=0)`.
Spectroscopy data put `acquisition_idx` first and keep only the dimensions
they use, for example `(acquisition_idx, read_idx, phase_idx, slice_idx,
repetition_idx, channel_idx)` for CSI.

## Coordinates

Non-dimension coordinates carry the physics:

- `x`, `y`, `z`: the scanner position of every voxel in mm, as written to
  DICOM ({doc}`coordinate_systems`).
- `time`: the start time of each repetition in s.
- `acquisition_time_s`: the time of each FID sample in s.

## Attributes

`attrs` record what the values are: `unit`, `domain`
(`"image"`, `"kspace"`, …), `signal_domain` for spectroscopy (`"temporal"` or
`"spectral"`), slice and read orientation, and the parameter files and
reconstruction they came from. Analysis functions read these instead of
guessing; a missing attribute is an error, not a silent default.

## Fit results

Fitting returns an {class}`xarray.Dataset` on the spatial grid of the input,
with the spectral dimension removed and a `metabolite` dimension added.
Variables are named `fit_amps`, `fit_freqs`, `fit_t2s`, …, and every result
holds a `fit_status` per voxel and the options it was computed with. The
{doc}`../tutorials/spectroscopy_fitting` tutorial shows one.

## Precision

Scanner images are stored as `float32`, raw reconstructions as `complex64`.
Calculations that need it convert to double precision internally; the stored
precision is a reconstruction option (`data_precision`).
