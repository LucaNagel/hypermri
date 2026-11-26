# Fourier convention

A raw reconstruction is only comparable with the scanner's own images if both
put every voxel at the same place. HyperMRI follows ParaVision's rule for
this, and every spatial Fourier transform in the package goes through the four
functions of {mod}`hypermri.reconstruction.centred_dft`.

## Read and phase axes

ParaVision places voxel $j$ of an axis with $n$ voxels at
$(j - n/2)\,\Delta x$ from the centre of the field of view. The centre is
therefore at index $n/2$: a voxel centre for even $n$, the border between the
two central voxels for odd $n$.

{func}`~hypermri.reconstruction.centred_dft.centered_ifft` maps k-space to the
image with the zero frequency at index `n // 2`:

$$
\text{image} = \operatorname{fftshift}\bigl(\operatorname{ifft}(\operatorname{ifftshift}(k))\bigr)
$$

For odd $n$ it first multiplies k-space by the half-voxel phase
$e^{-i\pi k/n}$, so that odd and even axes both end on the scanner grid.
{func}`~hypermri.reconstruction.centred_dft.centered_fft` is the inverse.
Without this phase, an odd-sized raw image lies half a voxel away from the
scanner image.

## Slice-encoding axis of 3D scans

The third axis of a 3D scan uses the *forward* FFT, as ParaVision does, so
raw slice $j$ is scanner slice $j$ without reversal
({func}`~hypermri.reconstruction.centred_dft.slice_encoding_to_image`). Its
centre is at index $(n-1)/2$, the middle of the symmetric slice grid; even
slice counts take a half-slice phase.

## What is not part of the transform

- **Orientation.** Flips and slice reversals of a scan orientation are applied
  after the transform, never inside it.
- **Field-of-view shifts.** An off-centre field of view is a linear phase on
  the k-space samples, applied before the transform.
- **Normalisation.** Read and phase use `norm="forward"`, the slice-encoding
  axis the default `"backward"`. Pass the same `norm` to the inverse function
  to restore the input exactly.

The {doc}`../tutorials/reconstruction` tutorial compares a raw reconstruction
with the scanner image voxel by voxel.
