# BrkRaw-DTI Hook

This hook provides support for Diffusion Tensor Imaging (DTI) scans.

## Features
- Automatic export of `.bvec` and `.bval` files.
- Gradient reorientation to RAS+ space.
- Live DTI calculation in BrkRaw Viewer.

## Parameters
- `PVM_DwNDiffDir`: Used to detect DTI scans.
- `PVM_DwGradVec`: Source of gradient vectors.
- `PVM_DwEffBval`: Source of b-values.
