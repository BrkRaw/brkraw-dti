import numpy as np
import logging
from typing import Any, Tuple, Optional, Union

logger = logging.getLogger("brkraw.dti")


def get_gradients(scan: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract b-values and b-vectors from Bruker scan parameters.
    
    Returns:
        bvals: (N,) array of b-values.
        bvecs: (N, 3) array of gradient vectors (normalized).
    """
    method = getattr(scan, "method", None)
    if not method:
        return None, None

    # --- 1. Resolve B-Values ---
    bvals = None
    if hasattr(method, "PVM_DwEffBval"):
        # Most modern PV versions
        bvals = np.array(method.PVM_DwEffBval)
    elif hasattr(method, "PVM_DwBvalEach"):
        bvals = np.array(method.PVM_DwBvalEach)
    
    # --- 2. Resolve Vectors (prefer candidates that match bvals) ---
    def _normalize_bvecs(raw: Any) -> Optional[np.ndarray]:
        if raw is None:
            return None
        arr = np.asarray(raw)
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                return None
            return arr.reshape(-1, 3)
        if arr.ndim >= 2:
            arr = np.squeeze(arr)
            if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
                return arr.T
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
        return None

    candidates: list[tuple[str, np.ndarray]] = []
    if hasattr(method, "PVM_DwGradVec"):
        cand = _normalize_bvecs(getattr(method, "PVM_DwGradVec"))
        if cand is not None:
            candidates.append(("PVM_DwGradVec", cand))
    if hasattr(method, "PVM_DwDir"):
        cand = _normalize_bvecs(getattr(method, "PVM_DwDir"))
        if cand is not None:
            candidates.append(("PVM_DwDir", cand))

    if not candidates:
        return None, None

    bvecs = None
    if bvals is not None:
        bvals_arr = np.asarray(bvals).reshape(-1)
        b0_mask = bvals_arr <= 50
        n_b0 = int(np.sum(b0_mask))
        n_dwi = int(np.sum(~b0_mask))
        best_score = -1
        best = None
        for name, cand in candidates:
            n = int(cand.shape[0])
            score = 0
            if n == len(bvals_arr):
                score = 3
            elif n_dwi > 0 and n == n_dwi:
                score = 2
            elif n_b0 > 0 and n == n_b0:
                score = 1
            if score > best_score or (score == best_score and best and n > best[1].shape[0]):
                best_score = score
                best = (name, cand)
        if best is not None:
            name, cand = best
            if n_dwi > 0 and cand.shape[0] == n_dwi:
                full = np.zeros((len(bvals_arr), 3), dtype=float)
                full[~b0_mask] = cand[:n_dwi]
                bvecs = full
            elif cand.shape[0] == len(bvals_arr):
                bvecs = cand
            elif n_b0 > 0 and cand.shape[0] == n_b0:
                logger.warning("B-vecs match only b0 count; diffusion directions missing.")
                full = np.zeros((len(bvals_arr), 3), dtype=float)
                full[b0_mask] = cand[:n_b0]
                bvecs = full
    if bvecs is None:
        # fallback: pick the largest candidate
        name, cand = max(candidates, key=lambda item: item[1].shape[0])
        bvecs = cand

    num_dirs = int(bvecs.shape[0])

    # Handle global b-value if per-direction b-value is missing
    if bvals is None:
        global_bval = getattr(method, "PVM_DwBVal", 0)
        bvals = np.ones(num_dirs) * global_bval
    else:
        bvals = np.asarray(bvals).reshape(-1)
        if bvals.size == 1 and num_dirs > 1:
            bvals = np.full(num_dirs, float(bvals[0]))

    # Ensure shapes match (fallback to truncation only if still mismatched)
    if len(bvals) != num_dirs:
        if num_dirs > len(bvals) and num_dirs % len(bvals) == 0:
            rep = int(num_dirs // len(bvals))
            bvals = np.repeat(bvals, rep, axis=0)
        if len(bvals) != num_dirs:
            logger.warning(
                "B-vals count (%s) != B-vecs count (%s). Truncating to minimum.",
                len(bvals),
                num_dirs,
            )
            min_len = min(len(bvals), num_dirs)
            bvals = bvals[:min_len]
            bvecs = bvecs[:min_len]

    # --- 4. Normalization ---
    # Normalize vectors to unit length (safely handling 0-vectors for b=0)
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    bvecs = bvecs / norms
    
    # Force bvec=0 where bval is effectively 0 (some protocols might have noise)
    # But usually keeping the direction is fine even for b=0. 
    # Let's trust the input direction unless it was explicitly zero.

    return bvals, bvecs


def reorient_gradients(
    scan: Any,
    bvecs: np.ndarray,
    *,
    affine: np.ndarray,
) -> np.ndarray:
    """
    Reorient gradient vectors from Logical (Read/Phase/Slice) to 
    Image/NIfTI Space (RAS) using the scan's affine transformation.
    
    This applies the rotation component of the affine matrix to the vectors.
    """
    # 1. Get the authoritative affine (usually Subject RAS)
    # Use explicit affine when provided (viewer RAS), otherwise query scan.
    if affine is None:
        try:
            affine = scan.get_affine()
        except Exception:
            affine = None
    if affine is None:
        logger.warning("No affine found for scan. Skipping gradient reorientation.")
        return bvecs

    # Handle multiple affines (multi-slice packs) - use the first one
    if isinstance(affine, tuple):
        affine = affine[0]

    # 2. Extract Rotation Matrix (3x3)
    # R = M @ S^-1, but for simple rigid/scaled affines, 
    # we can just normalize the columns of the upper-left 3x3.
    M = affine[:3, :3]
    
    # Calculate column norms (scales)
    scales = np.linalg.norm(M, axis=0)
    
    # Avoid div by zero
    scales[scales == 0] = 1.0
    
    # Rotation matrix (approximate if sheared, exact if rigid+scaled)
    R = M / scales

    # 3. Apply Rotation: v_new = R @ v_old
    # bvecs is (N, 3), so we transpose: (R @ bvecs.T).T
    rotated_bvecs = (R @ bvecs.T).T
    
    # 4. Normalize again to be safe
    norms = np.linalg.norm(rotated_bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    rotated_bvecs = rotated_bvecs / norms
    
    return rotated_bvecs


def fit_tensor_ols(
    data: np.ndarray, 
    bvals: np.ndarray, 
    bvecs: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit DTI tensor using Ordinary Least Squares (OLS).
    
    Args:
        data: (X, Y, Z, N) diffusion weighted data.
        bvals: (N,) b-values.
        bvecs: (N, 3) gradient vectors.
        mask: (X, Y, Z) brain mask (boolean). If None, fits all voxels.
    
    Returns:
        evals: (X, Y, Z, 3) eigenvalues.
        evecs: (X, Y, Z, 3, 3) eigenvectors.
    """
    # Simple OLS implementation: ln(S/S0) = -b * g^T D g
    # D is symmetric 3x3 tensor -> 6 unique elements: Dxx, Dyy, Dzz, Dxy, Dxz, Dyz
    
    if mask is not None:
        # Flatten mask to verify valid voxels exist
        if not np.any(mask):
            return None, None
            
    # Prepare data
    data = data.astype(np.float64)
    data[data <= 0] = 1e-6  # Avoid log(0) or negative signal
    log_s = np.log(data)    # (X, Y, Z, N)
    
    # Design matrix B: N x 7 (including B0 intercept)
    # [1, -b*gx^2, -b*gy^2, -b*gz^2, -2*b*gx*gy, -2*b*gx*gz, -2*b*gy*gz]
    N = len(bvals)
    X = np.zeros((N, 7))
    X[:, 0] = 1.0 # Intercept (ln(S0))
    
    bx, by, bz = bvecs.T
    X[:, 1] = -bvals * bx**2
    X[:, 2] = -bvals * by**2
    X[:, 3] = -bvals * bz**2
    X[:, 4] = -2 * bvals * bx * by
    X[:, 5] = -2 * bvals * bx * bz
    X[:, 6] = -2 * bvals * by * bz
    
    # Solve X * beta = log_s
    # We use Pseudo-Inverse: beta = pinv(X) @ Y
    # Shape of X: (N, 7)
    # Shape of Y needs to be (N, Voxels)
    
    try:
        X_pinv = np.linalg.pinv(X) # (7, N)
    except np.linalg.LinAlgError:
        logger.error("DTI OLS fit failed: Singular Matrix.")
        return None, None
        
    spatial_shape = data.shape[:3]
    num_voxels = np.prod(spatial_shape)
    
    # Reshape Data: (Voxels, N) -> Transpose for matmul -> (N, Voxels)
    # NOTE: If we use a mask, we only process masked voxels to save time
    
    if mask is not None:
        # Extract only valid voxels: (M, N) where M is num_masked
        # mask shape: (X, Y, Z)
        Y_masked = log_s[mask] # (M, N)
        Y_target = Y_masked.T  # (N, M)
    else:
        Y_target = log_s.reshape(num_voxels, N).T # (N, Voxels)
        
    # Batch Matmul
    # beta: (7, M) or (7, Voxels)
    beta = X_pinv @ Y_target
    
    # Map back to full grid
    if mask is not None:
        # Initialize full grid with zeros
        D_flat = np.zeros((num_voxels, 3, 3))
        # We need indices to map back. 
        # Easier to create a full beta grid first
        beta_full = np.zeros((7, num_voxels))
        beta_full[:, mask.reshape(-1)] = beta
        beta = beta_full
        
    # Dxx=beta[1], Dyy=beta[2], Dzz=beta[3], Dxy=beta[4], Dxz=beta[5], Dyz=beta[6]
    D = np.zeros((num_voxels, 3, 3))
    D[:, 0, 0] = beta[1]
    D[:, 1, 1] = beta[2]
    D[:, 2, 2] = beta[3]
    D[:, 0, 1] = D[:, 1, 0] = beta[4]
    D[:, 0, 2] = D[:, 2, 0] = beta[5]
    D[:, 1, 2] = D[:, 2, 1] = beta[6]
    
    # Eigendecomposition
    # eigh handles symmetric matrices and is faster
    evals, evecs = np.linalg.eigh(D)
    
    # Reshape back to image dimensions
    evals = evals.reshape(*spatial_shape, 3)
    evecs = evecs.reshape(*spatial_shape, 3, 3)
    
    # Sort eigenvalues (descending)
    # eigh returns ascending, so reverse
    evals = evals[..., ::-1]
    evecs = evecs[..., ::-1]
    
    # Zero out background if mask provided (cleanup)
    if mask is not None:
        evals[~mask] = 0
        evecs[~mask] = 0
        
    return evals, evecs


def compute_fa(evals: np.ndarray) -> np.ndarray:
    """Compute Fractional Anisotropy."""
    l1, l2, l3 = evals[..., 0], evals[..., 1], evals[..., 2]
    # Clip negative eigenvalues
    l1 = np.maximum(l1, 1e-9)
    l2 = np.maximum(l2, 1e-9)
    l3 = np.maximum(l3, 1e-9)
    
    md = (l1 + l2 + l3) / 3.0
    
    num = (l1 - md)**2 + (l2 - md)**2 + (l3 - md)**2
    den = l1**2 + l2**2 + l3**2
    
    fa = np.sqrt(1.5 * num / (den + 1e-9))
    return np.clip(fa, 0, 1)


def compute_md(evals: np.ndarray) -> np.ndarray:
    """Compute Mean Diffusivity."""
    return np.mean(evals, axis=-1)


def compute_color_fa(fa: np.ndarray, evecs: np.ndarray) -> np.ndarray:
    """Compute RGB Color FA map."""
    # Primary eigenvector (v1) corresponds to largest eigenvalue (index 0)
    v1 = evecs[..., 0] 
    # Color = FA * |v1|
    # Result is (X, Y, Z, 3)
    return fa[..., None] * np.abs(v1)


def compute_ad(evals: np.ndarray) -> np.ndarray:
    """Compute Axial Diffusivity (AD)."""
    return evals[..., 0]


def compute_rd(evals: np.ndarray) -> np.ndarray:
    """Compute Radial Diffusivity (RD)."""
    return np.mean(evals[..., 1:3], axis=-1)


def compute_trace(evals: np.ndarray) -> np.ndarray:
    """Compute Trace (sum of eigenvalues)."""
    return np.sum(evals, axis=-1)
