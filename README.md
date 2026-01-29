# BrkRaw DTI Extension

This hook provides **Diffusion Tensor Imaging (DTI)** support for BrkRaw.

## Features

1.  **DTI Conversion**:
    *   Automatically exports `.bval` and `.bvec` (FSL format) files alongside the NIfTI image.
    *   Correctly reorients gradient vectors (`bvecs`) from the logical frame to the image frame (RAS+).

2.  **DTI Viewer & QC**:
    *   Adds a **DTI Tab** to the BrkRaw Viewer.
    *   **Calculate DTI**: Performs real-time Tensor fitting (OLS).
    *   **Visualization**: Displays Fractional Anisotropy (FA), Mean Diffusivity (MD), and **Directionally Encoded Color FA (DEC-FA)** maps directly in the viewer.
    *   **DEC-FA QC**: Allows immediate verification of gradient orientation (Red: L-R, Green: A-P, Blue: S-I).

## Usage

### Conversion
The hook is automatically selected for scans with diffusion directions (`PVM_DwNDiffDir > 0`).

```bash
brkraw convert /path/to/study -s 10
```

### Viewer
Open the viewer and select a DTI scan. The "DTI" tab will appear in the control panel.

1.  Click **Calculate DTI**.
2.  Wait for the calculation to finish.
3.  Use the dropdown to switch between `FA`, `MD`, `DEC-FA`, and `b0` views.

## Installation

```bash
pip install .
brkraw hook install dti
```