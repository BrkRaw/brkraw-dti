import logging
from pathlib import Path
from types import MethodType
from typing import Optional, Union, Tuple, Any, Iterable

import numpy as np
from nibabel.nifti1 import Nifti1Image

from brkraw.apps.loader.types import ToFilename

from .logic import get_gradients, reorient_gradients

logger = logging.getLogger("brkraw.dti")


def _strip_nii_suffix(path: Union[str, Path]) -> str:
    p = Path(str(path))
    name = p.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return p.stem


def _write_bvec_bval(filename: Union[str, Path], bvals: Optional[np.ndarray], bvecs: Optional[np.ndarray]) -> None:
    if bvals is None and bvecs is None:
        return
    p = Path(str(filename))
    stem = _strip_nii_suffix(p)
    parent = p.parent
    if bvals is not None:
        bval_path = parent / f"{stem}.bval"
        logger.debug("Saving bvals to %s", bval_path)
        np.savetxt(bval_path, np.asarray(bvals).reshape(1, -1), fmt="%.0f")
    if bvecs is not None:
        bvec_path = parent / f"{stem}.bvec"
        logger.debug("Saving bvecs to %s", bvec_path)
        np.savetxt(bvec_path, np.asarray(bvecs).T, fmt="%.6f")


def _attach_bvec_bval_writer(
    nii: Nifti1Image,
    bvals: Optional[np.ndarray],
    bvecs: Optional[np.ndarray],
) -> Nifti1Image:
    if bvals is None and bvecs is None:
        return nii
    orig_to_filename = nii.to_filename

    def _to_filename(self: Nifti1Image, filename: Any, *args: Any, **kwargs: Any) -> Any:
        result = orig_to_filename(str(filename), *args, **kwargs)
        _write_bvec_bval(filename, bvals, bvecs)
        return result

    nii.to_filename = MethodType(_to_filename, nii)
    return nii


def _normalize_sequence(value: Union[np.ndarray, Tuple[np.ndarray, ...]]) -> Tuple[np.ndarray, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _resolve_reco_id_from_affines(scan: Any, affines: Iterable[np.ndarray]) -> Optional[int]:
    affine_info = getattr(scan, "affine_info", None)
    if not isinstance(affine_info, dict):
        return None
    aff_list = [np.asarray(a) for a in affines]
    for reco_id, info in affine_info.items():
        if not isinstance(info, dict):
            continue
        ref_affines = info.get("affines")
        if ref_affines is None:
            continue
        if isinstance(ref_affines, tuple):
            ref_list = list(ref_affines)
        else:
            ref_list = [ref_affines]
        if len(ref_list) != len(aff_list):
            continue
        matched = True
        for ref, cand in zip(ref_list, aff_list):
            try:
                if not np.allclose(np.asarray(ref), np.asarray(cand), rtol=1e-4, atol=1e-4):
                    matched = False
                    break
            except Exception:
                matched = False
                break
        if matched:
            try:
                return int(reco_id)
            except Exception:
                return None
    return None


def _apply_dti_units(nii: Nifti1Image, *, xyz_units: Optional[str]) -> None:
    xyz_unit = xyz_units
    if xyz_unit is None:
        try:
            xyz_unit = nii.header.get_xyzt_units()[0]
        except Exception:
            xyz_unit = None
    if xyz_unit is None:
        xyz_unit = "mm"
    try:
        nii.header.set_xyzt_units(xyz_unit, "unknown")
    except Exception:
        try:
            nii.header.set_xyzt_units(xyz_unit, "sec")
        except Exception:
            pass


def convert(
    scan: Any,
    dataobj: Union[np.ndarray, Tuple[np.ndarray, ...]],
    affine: Union[np.ndarray, Tuple[np.ndarray, ...]],
    **kwargs: Any,
) -> Optional[Union[ToFilename, Tuple[ToFilename, ...]]]:
    """
    DTI conversion: use the default NIfTI creation path when possible,
    and add FSL bvec/bval sidecars on save.
    """
    bvals, bvecs = get_gradients(scan)

    if bvecs is not None:
        # Conversion keeps image axes in the original acquisition space;
        # do not reorient gradients for file export.
        try:
            norms = np.linalg.norm(np.asarray(bvecs), axis=1)
            if np.all(norms <= 0):
                logger.warning("DTI bvecs are all zeros after parsing; check gradient parameters.")
        except Exception:
            logger.warning("DTI bvecs validation failed; check gradient parameters.")

    dataobjs = list(_normalize_sequence(dataobj))
    affines = list(_normalize_sequence(affine))
    if len(affines) == 1 and len(dataobjs) > 1:
        affines = affines * len(dataobjs)

    xyz_units = kwargs.get("xyz_units", "mm") or "mm"
    t_units = kwargs.get("t_units", "sec")
    override_header = kwargs.get("override_header", None)

    reco_id = _resolve_reco_id_from_affines(scan, affines)
    results: list[Nifti1Image] = []

    niiobjs: Optional[Union[Nifti1Image, Tuple[Nifti1Image, ...]]] = None
    if reco_id is not None and hasattr(scan, "get_nifti1image"):
        try:
            niiobjs = scan.get_nifti1image(
                reco_id=reco_id,
                dataobjs=tuple(dataobjs),
                affines=tuple(affines),
                override_header=override_header,
                xyz_units=xyz_units,
                t_units=t_units,
            )
        except Exception:
            niiobjs = None

    if niiobjs is None:
        for d, aff in zip(dataobjs, affines):
            nii = Nifti1Image(d, aff)
            results.append(nii)
    else:
        if isinstance(niiobjs, tuple):
            results.extend(list(niiobjs))
        else:
            results.append(niiobjs)

    for nii in results:
        _apply_dti_units(nii, xyz_units=xyz_units)
        _attach_bvec_bval_writer(nii, bvals, bvecs)

    if len(results) == 1:
        return results[0]
    return tuple(results)


# HOOK export
HOOK = {
    # We use core's get_dataobj and get_affine (default behavior)
    # but override the final conversion step.
    "convert": convert
}
