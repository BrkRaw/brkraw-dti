# BrkRaw DTI Extension

This hook adds **Diffusion Tensor Imaging (DTI)** support to **BrkRaw**, providing both conversion and quality-control tools for diffusion datasets.

## Features

### 1. DTI Conversion
- Automatically exports .bval and .bvec files (FSL format) alongside the converted NIfTI image.
- Correctly reorients diffusion gradient vectors (bvecs) from the logical acquisition frame to the image frame (RAS+) using the affine.

### 2. DTI Viewer & Quality Control
- Adds a dedicated **DTI** tab to the BrkRaw Viewer.
- **DTI Calculation**: Performs real-time tensor fitting using Ordinary Least Squares (OLS).
- **Visualization**:
  - Fractional Anisotropy (FA)
  - Mean Diffusivity (MD)
  - Directionally Encoded Color FA (DEC-FA)
- **DEC-FA QC**: Enables immediate visual verification of gradient orientation:
  - Red: Left-Right (L-R)
  - Green: Anterior-Posterior (A-P)
  - Blue: Superior-Inferior (S-I)

## Usage

### Conversion
The hook is automatically applied to scans with diffusion directions (PVM_DwNDiffDir > 0).

```bash
brkraw convert /path/to/study -s 10
```

### Viewer
Open the BrkRaw Viewer and select a DTI scan. A **DTI** tab will appear under the Extensions panel.

Tip: Right-click the extensions tab to detach it into a separate window for easier inspection.

1. Click **Calculate DTI**.
2. Wait for the tensor fitting to complete.
3. Use the dropdown menu to switch between FA, MD, DEC-FA, and b0 views.

## Installation

```bash
pip install .
brkraw hook install dti
```
