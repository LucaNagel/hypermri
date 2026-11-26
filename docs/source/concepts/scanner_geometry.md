# From read/phase/slice to x/y/z

A reconstructed image is stored on its acquisition grid: `read_idx`,
`phase_idx` and `slice_idx`. Two scans of the same animal usually have
different grids, for example a coronal CSI with 8 × 8 voxels and an axial
anatomy with 256 × 256 pixels. To draw one on top of the other, HyperMRI
needs to know where every voxel lies in the room. This page explains how a
Bruker scan gets that position, and how the viewer uses it to put the right
anatomy slice under a CSI map.

## The short answer

- Every Bruker image carries `x`, `y` and `z` coordinates in mm for each
  voxel, next to its index dimensions.
- They are computed from the ParaVision `visu_pars` file: `VisuCorePosition`,
  `VisuCoreOrientation`, `VisuCoreSize` and `VisuCoreExtent`, together with
  `RECO_transposition` from the `reco` file.
- These are the same numbers ParaVision writes into its DICOM export
  (`Image Position (Patient)` and `Image Orientation (Patient)`). The x/y/z
  coordinates therefore agree with what a DICOM viewer shows.
- The frame is the DICOM patient frame: `x` grows towards the patient's left,
  `y` towards the back (posterior) and `z` towards the head. Patient position
  and orientation (head first, prone, ...) are already included by
  ParaVision.
- The viewer finds the anatomy slice that lies closest to the displayed CSI
  plane and draws both on the same mm axes. Nothing is resampled.

```{mermaid}
flowchart LR
    idx["voxel index<br/>read_idx, phase_idx, slice_idx"]
    tr{{"RECO_transposition<br/>read along columns or rows?"}}
    cr["column, row, frame<br/>of the stored 2dseq image"]
    visu["VisuCorePosition<br/>VisuCoreOrientation<br/>VisuCoreExtent / VisuCoreSize"]
    xyz["x, y, z in mm<br/>(DICOM patient frame)"]
    idx --> tr --> cr --> xyz
    visu --> xyz
```

## Step 1: which index runs along which image axis?

ParaVision stores each slice (a *frame*) as a 2D image with columns and rows.
The read direction normally runs along the columns. Some protocols store the
image transposed, with phase along the columns; `RECO_transposition` records
this per frame:

| `RECO_transposition` | along the columns (axis **u**) | along the rows (axis **v**) |
|---|---|---|
| 0 | `read_idx` | `phase_idx` |
| 1 | `phase_idx` | `read_idx` |

HyperMRI keeps `read_idx` and `phase_idx` as dimension names in both cases.
Only the mapping to the stored image axes changes.

## Step 2: from column and row to mm

`VisuCoreOrientation` holds three unit vectors per frame: the column
direction **u**, the row direction **v** and the slice normal **n**.
`VisuCorePosition` holds the centre of the first voxel of each frame, and
`VisuCoreExtent / VisuCoreSize` gives the voxel spacing along **u** and **v**.
A voxel in column $c$ and row $r$ of frame $k$ then lies at

$$
\begin{pmatrix} x \\ y \\ z \end{pmatrix}
= \mathbf{P}_k + c\,\Delta u\,\mathbf{u} + r\,\Delta v\,\mathbf{v},
\qquad
\Delta u = \frac{\text{Extent}_0}{\text{Size}_0},\;
\Delta v = \frac{\text{Extent}_1}{\text{Size}_1}.
$$

- **2D multi-slice:** `VisuCorePosition` lists one $\mathbf{P}_k$ for every
  slice, so slice gaps and the stored slice order are already included.
- **3D volume:** `VisuCorePosition` lists only one position $\mathbf{P}$.
  Slice $k$ then lies at $\mathbf{P} + k\,\Delta n\,\mathbf{n}$, where
  $\Delta n$ is the slice spacing.

Try it below: pick an orientation, switch the transposition, and click a
voxel. The page repeats the calculation of
{func}`~hypermri.utils.coordinates.grids.get_xyz_grid_from_visu` for a small
synthetic scan and shows every number that goes into it.

<iframe class="hm-snapshot hm-explainer" src="../_static/explainers/voxel_to_xyz.html"
        loading="lazy" title="Voxel index to x/y/z"></iframe>

## Where the coordinates live

Every reconstruction of a Bruker scan gets the coordinates attached in the
same place, whether it is the scanner image (`"seq2d"`) or HyperMRI's own
reconstruction from raw data (`"raw"`):

```python
image = scan.scanner_data           # or scan.data, scan.primary_data
image.x                             # 3D grid: (read_idx, phase_idx, slice_idx), mm
image.x.isel(read_idx=5, phase_idx=1, slice_idx=2)   # one voxel
image.read_idx                      # still the index axis used for selection
```

- `x`, `y` and `z` are *auxiliary* coordinates: each has the three spatial
  dimensions. They are not separate axes, so `image.sel(x=3.0)` does not work.
  Use `isel` on the index dimensions to select.
- The raw reconstruction lands on the same voxel grid as the scanner image
  (see {doc}`fourier_convention`). Before it is stored, HyperMRI checks that
  both have the same sizes, mm coordinates and orientation, and refuses to
  store it otherwise.
- If a scan has no usable geometry, for example because `visu_pars` is
  missing, it keeps plain index axes and has no `x`, `y` and `z`. The same
  happens for a ParaVision reconstruction whose geometry differs from the
  acquisition, and for scans with several slice packages.

Besides x/y/z, the images also carry `read`, `phase` and `slice` positions in
mm along the acquisition axes. They describe distances inside one scan;
comparing scans needs x/y/z.

## How the viewer puts an anatomy under a CSI map

`csi.plot(anatomy=anatomy)` shows a CSI map on top of an anatomical image.
The viewer does this with the x/y/z coordinates of both scans:

1. **Find the plane normal.** Over the displayed CSI plane, one of x, y and z
   barely changes. That coordinate is the normal, and its median value is the
   plane position.
2. **Find the matching anatomy axis.** In the anatomy, find the dimension
   along which that coordinate changes. For an anatomy with the same
   orientation this is `slice_idx`. For an anatomy acquired in another
   orientation it can be `read_idx` or `phase_idx`, which gives a
   reformatted view of a 3D volume.
3. **Pick the nearest slice.** Take the anatomy slice whose position is
   closest to the CSI plane. If the CSI plane lies outside the anatomy's
   slices, the viewer warns and can hide the background.
4. **Draw on common axes.** Both images are placed by the outer edges of their
   voxels in mm (centre ± half a voxel). The browser draws them on the same
   x/z (or x/y, y/z) axes, so a 4 mm CSI voxel covers exactly the anatomy
   pixels inside it.

<iframe class="hm-snapshot hm-explainer" src="../_static/explainers/anatomy_matching.html"
        loading="lazy" title="Anatomy under a CSI map"></iframe>

Two consequences follow:

- The anatomy is never interpolated between slices. A thick CSI slice is
  shown over one thin anatomy slice, the one at its centre.
- The method is exact for planes parallel to the x, y and z axes. For oblique
  slices, the viewer still picks the nearest slice and uses the bounding box
  of the voxel centres, which is an approximation.

The same coordinates are used to transfer masks from one scan to another with
{meth}`~hypermri.loading.experiment.MRIExperiment.apply_mask_to`.

## See also

- {doc}`coordinate_systems` for a short reference of both coordinate frames.
- {doc}`fourier_convention` for why the raw reconstruction lands on the
  scanner's voxel grid.
- {doc}`viewer` for what runs in Python and what in the browser.
