# Reconstruction

A scan can be reconstructed in two ways, called *reconstruction types*:

| | `"seq2d"` | `"raw"` |
| --- | --- | --- |
| Source | the scanner image (Bruker `2dseq`) | the acquired data (`fid`, `ser`, `rawdata.job0`) |
| Fourier transform | none, the scanner did it | by HyperMRI, where the sequence needs it |
| Values | real (usually magnitude) | complex, phase kept |
| Stored on the scan | `scan.scanner_data` | `scan.data` |

Both results carry the same labelled dimensions and scanner coordinates
({doc}`data_model`, {doc}`coordinate_systems`), so they can be compared voxel by
voxel. The {doc}`../tutorials/reconstruction` tutorial does this for a FLASH scan.

## Which code runs

{func}`~hypermri.reconstruction.selection.find_reconstruction` looks up the
strategy class and its default options in one Python table,
{data}`~hypermri.reconstruction.selection.RECONSTRUCTIONS`. It tries the
scan's method name, then its canonical sequence ID, then the vendor's default
entry; there is no other fallback. `scan.reconstruct(kind, **options)` merges
the defaults with the given options, rejects options the strategy does not
accept and runs it.

Strategies live in `hypermri.reconstruction.strategies`, one module per vendor
and sequence family. Imaging strategies sort the acquired samples into k-space
and transform the spatial axes with the centred DFT of
{doc}`fourier_convention`. Spectroscopy strategies keep the FID along the
acquisition axis.

## When it runs

A sequence class declares which reconstructions each loading mode runs, as
`RECONSTRUCTION_POLICY`:

| Sequence family | `"lazy"` | `"minimal"` | `"full"` |
| --- | --- | --- | --- |
| anatomy (FLASH, RARE, …), CSI | none | `seq2d` | `seq2d`, `raw` |
| single-voxel spectroscopy | none | `raw` | `raw` |

Constructing a scan never reconstructs; the loading code applies the policy
afterwards. Reading `scan.primary_data` may still start a raw reconstruction
on demand. Errors of a reconstruction are raised, not hidden.

## Raw data stay untouched

A reconstruction never changes the acquired data held by the vendor handler,
so running it again gives the same result. For Bruker CSI, the leading
digital-filter samples are removed once from the sorted result, and the time
and frequency axes are rebuilt for the remaining samples: the processed FID
starts at time zero.

## Vendors

Bruker ParaVision is the mature path. Siemens TWIX loading and Agilent/Varian,
Magritek and Bruker TopSpin spectroscopy loading are experimental; the
{doc}`../support` page lists what is supported.
