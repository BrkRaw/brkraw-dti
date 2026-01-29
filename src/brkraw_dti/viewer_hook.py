from __future__ import annotations

import logging
import threading
from typing import cast, Any, Optional
from pathlib import Path
import re
import sys

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from nibabel.nifti1 import Nifti1Image

from .logic import (
    compute_color_fa,
    compute_fa,
    compute_md,
    compute_ad,
    compute_rd,
    compute_trace,
    fit_tensor_ols,
    get_gradients,
    reorient_gradients,
)

logger = logging.getLogger("brkraw.dti.viewer")


class DTIViewerHook:
    name = "brkraw-dti"
    tab_title = "DTI"
    priority = 10

    def __init__(self) -> None:
        self._panel: Optional[DTIViewerPanel] = None

    def build_tab(self, parent: tk.Misc, app: Any) -> Optional[tk.Widget]:
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        wrapper = ttk.Frame(frame)
        wrapper.grid(row=0, column=0, sticky="nsew")
        wrapper.grid_propagate(True)
        wrapper.columnconfigure(0, weight=1)
        wrapper.rowconfigure(0, weight=1)
        panel = DTIViewerPanel(wrapper, app=app)
        panel.grid(row=0, column=0, sticky="n")
        panel.refresh_from_viewer()
        register = getattr(app, "register_viewer_space_listener", None)
        if callable(register):
            try:
                register(panel.on_viewer_space_change)
            except Exception:
                pass
        self._panel = panel
        return frame

    def on_dataset_loaded(self, app: Any) -> None:
        if self._panel is not None:
            self._panel.refresh_from_viewer()

    def on_scan_selected(self, app: Any) -> None:
        if self._panel is not None:
            self._panel.refresh_from_viewer()


class DTIViewerPanel(ttk.Frame):
    def __init__(self, parent: tk.Misc, *, app: Any) -> None:
        super().__init__(parent)
        self._app = app
        self._maps: dict[str, np.ndarray] = {}
        self._bvals: Optional[np.ndarray] = None
        self._bvecs_ras: Optional[np.ndarray] = None
        self._calc_thread: Optional[threading.Thread] = None
        self._viewer_cache: dict[str, Any] = {}
        self._last_space: Optional[str] = None

        self._status_var = tk.StringVar(value="")
        self._info_var = tk.StringVar(value="")
        self._map_var = tk.StringVar(value="DWI")

        self._calc_button: Optional[ttk.Button] = None
        self._map_combo: Optional[ttk.Combobox] = None
        self._reset_button: Optional[ttk.Button] = None
        self._save_button: Optional[ttk.Button] = None

        self._make_ui()

    def _make_ui(self) -> None:
        self.columnconfigure(0, weight=1)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="DTI Maps").grid(row=0, column=0, sticky="n")

        controls = ttk.Frame(self)
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        self._calc_button = ttk.Button(controls, text="Calculate DTI", command=self._start_calculation, width=14)
        self._calc_button.grid(row=0, column=0, sticky="e", padx=(0, 10))

        self._reset_button = ttk.Button(controls, text="Reset", command=self._reset_panel, width=10)
        self._reset_button.grid(row=0, column=1, sticky="w", padx=(10, 0))

        ttk.Label(controls, text="View").grid(row=1, column=0, sticky="e", pady=(6, 0), padx=(0, 6))
        self._map_combo = ttk.Combobox(
            controls,
            textvariable=self._map_var,
            state="readonly",
            values=["DWI"],
            width=18,
        )
        self._map_combo.grid(row=1, column=1, sticky="w", pady=(6, 0))
        self._map_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_map())

        info = ttk.Label(self, textvariable=self._info_var, anchor="center")
        info.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        status_row = ttk.Frame(self)
        status_row.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        status_row.columnconfigure(0, weight=1)
        status_row.columnconfigure(1, weight=1)
        status = ttk.Label(status_row, textvariable=self._status_var, anchor="e")
        status.grid(row=0, column=0, sticky="e")

        self._save_button = ttk.Button(
            status_row,
            text="Save Maps",
            command=self._save_maps,
            width=12,
            state="disabled",
        )
        self._save_button.grid(row=0, column=1, sticky="w", padx=(10, 0))

    def refresh_from_viewer(self) -> None:
        # Reset internal DTI state (viewer controller owns hook state on scan/dataset change).
        self._restore_viewer(unlock_hook=False)
        self._maps = {}
        self._bvals = None
        self._bvecs_ras = None
        self._viewer_cache = {}
        self._map_var.set("DWI")
        if self._map_combo is not None:
            self._map_combo["values"] = ["DWI"]
        if self._save_button is not None:
            self._save_button.configure(state="disabled")
        self._info_var.set("")
        self._last_space = None
        scan = self._resolve_scan_from_app()
        data = self._resolve_viewer_volume()
        if scan is None:
            self._set_status("No scan selected.")
            self._info_var.set("")
            return
        bvals, bvecs = get_gradients(scan)
        if bvals is None or bvecs is None or len(bvals) < 2:
            self._set_status("DTI gradients not detected.")
            self._info_var.set("")
            return
        affine = cast(np.ndarray, self._resolve_viewer_affine())
        bvecs_ras = reorient_gradients(scan, bvecs, affine=affine)
        self._bvals = bvals
        self._bvecs_ras = bvecs_ras
        shape_text = ""
        if isinstance(data, np.ndarray):
            shape_text = f"Data shape: {tuple(data.shape)}"
        bvals_arr = np.asarray(bvals, dtype=float)
        bvals_arr = np.nan_to_num(bvals_arr, nan=0.0, posinf=0.0, neginf=0.0)
        dir_count = 0
        b0_count = 0
        if self._bvecs_ras is not None:
            try:
                norms = np.linalg.norm(self._bvecs_ras, axis=1)
                dir_count = int(np.sum(norms > 0))
            except Exception:
                dir_count = 0
        if dir_count == 0:
            dir_count = int(np.sum(bvals_arr > 50))
        if dir_count == 0:
            try:
                method = getattr(scan, "method", None)
                ndiff = getattr(method, "PVM_DwNDiffDir", None) if method else None
                if ndiff is not None:
                    dir_count = int(np.asarray(ndiff).reshape(-1)[0])
            except Exception:
                dir_count = 0
        if dir_count > 0:
            b0_count = max(int(len(bvals_arr)) - dir_count, 0)
        else:
            b0_count = int(np.sum(bvals_arr <= 50))
        self._info_var.set(
            f"Dirs: {dir_count}  b0: {b0_count}  {shape_text}"
        )
        self._set_status("Ready.")
        self._last_space = self._read_viewer_space()

    def _read_viewer_space(self) -> Optional[str]:
        app = self._app
        if app is None:
            return None
        state = getattr(app, "state", None)
        viewer_state = getattr(state, "viewer", None) if state is not None else None
        if viewer_state is None:
            return None
        try:
            return str(getattr(viewer_state, "space", None) or "")
        except Exception:
            return None

    def _read_viewer_flip(self) -> tuple[bool, bool, bool]:
        app = self._app
        if app is None:
            return (False, False, False)
        state = getattr(app, "state", None)
        viewer_state = getattr(state, "viewer", None) if state is not None else None
        if viewer_state is None:
            return (False, False, False)
        return (
            bool(getattr(viewer_state, "flip_x", False)),
            bool(getattr(viewer_state, "flip_y", False)),
            bool(getattr(viewer_state, "flip_z", False)),
        )

    def _apply_flip_to_data_affine(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        *,
        flip_x: bool,
        flip_y: bool,
        flip_z: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(data)
        if arr.ndim < 3:
            return arr, affine
        dims = arr.shape[:3]
        out_affine = np.asarray(affine, dtype=float)
        for axis, flag in enumerate((flip_x, flip_y, flip_z)):
            if not flag:
                continue
            arr = np.flip(arr, axis=axis)
            flip_mat = np.eye(4, dtype=float)
            flip_mat[axis, axis] = -1.0
            flip_mat[axis, 3] = float(dims[axis] - 1)
            out_affine = out_affine @ flip_mat
        arr = np.ascontiguousarray(arr)
        return arr, out_affine

    def on_viewer_space_change(self, value: str) -> None:
        new_space = str(value or "")
        if self._last_space == new_space:
            return
        self._last_space = new_space
        if self._maps:
            self._reset_panel()
        else:
            self._set_status("Space changed.")

    def _start_calculation(self) -> None:
        if self._calc_thread is not None and self._calc_thread.is_alive():
            return
        self._lock_viewer_hook(True)
        if self._calc_button is not None:
            self._calc_button.configure(state="disabled")
        self._set_status("Calculating...")
        self._calc_thread = threading.Thread(target=self._do_calculation, daemon=True)
        self._calc_thread.start()

    def _do_calculation(self) -> None:
        scan = self._resolve_scan_from_app()
        data = self._resolve_dwi_volume()
        if scan is None:
            self._schedule_status("No scan selected.")
            self._schedule_calc_ready()
            return
        if data is None or data.ndim != 4:
            self._schedule_status("Need full 4D DWI volume. Enable Viewer Hook if needed.")
            self._schedule_calc_ready()
            return
        if self._bvals is None or self._bvecs_ras is None:
            self._schedule_status("DTI gradients not available.")
            self._schedule_calc_ready()
            return

        bvals = np.asarray(self._bvals).astype(float)
        bvecs = np.asarray(self._bvecs_ras).astype(float)

        if data.shape[3] != len(bvals):
            count = min(data.shape[3], len(bvals))
            data = data[..., :count]
            bvals = bvals[:count]
            bvecs = bvecs[:count]

        b0_mask = bvals <= 50
        if not np.any(b0_mask):
            b0 = data[..., 0]
        else:
            b0 = np.mean(data[..., b0_mask], axis=3)

        mask = None

        evals, evecs = fit_tensor_ols(data, bvals, bvecs, mask=mask)
        if evals is None or evecs is None:
            self._schedule_status("DTI fit failed.")
            self._schedule_calc_ready()
            return

        fa = compute_fa(evals)
        md = compute_md(evals)
        ad = compute_ad(evals)
        rd = compute_rd(evals)
        trace = compute_trace(evals)
        cfa = compute_color_fa(fa, evecs)
        fa = np.nan_to_num(fa, nan=0.0, posinf=0.0, neginf=0.0)
        md = np.nan_to_num(md, nan=0.0, posinf=0.0, neginf=0.0)
        ad = np.nan_to_num(ad, nan=0.0, posinf=0.0, neginf=0.0)
        rd = np.nan_to_num(rd, nan=0.0, posinf=0.0, neginf=0.0)
        trace = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)
        cfa = np.nan_to_num(cfa, nan=0.0, posinf=0.0, neginf=0.0)
        cfa = np.clip(cfa, 0.0, 1.0)

        self._maps = {
            "b0": b0,
            "FA": fa,
            "MD": md,
            "AD": ad,
            "RD": rd,
            "TRACE (3xMD)": trace,
            "DEC-FA": cfa,
        }

        self._schedule_finalize()

    def _apply_map(self) -> None:
        name = self._map_var.get() or "DWI"
        if name == "DWI":
            self._restore_viewer()
            return
        data = self._maps.get(name)
        if data is None:
            return
        app = self._app
        if app is None:
            return
        if not self._viewer_cache:
            self._viewer_cache = {
                "volume": getattr(app, "_viewer_volume", None),
                "frames": getattr(app, "_viewer_frames", None),
            }
        setattr(app, "_viewer_volume", data)
        try:
            frames = int(data.shape[3]) if data.ndim >= 4 else 1
            setattr(app, "_viewer_frames", frames)
            state = getattr(app, "state", None)
            if state is not None:
                viewer_state = getattr(state, "viewer", None)
                if viewer_state is not None:
                    viewer_state.frame_index = 0
        except Exception:
            pass
        self._request_render()

    def _restore_viewer(self, *, unlock_hook: bool = False) -> None:
        if not self._viewer_cache:
            return
        app = self._app
        if app is None:
            return
        setattr(app, "_viewer_volume", self._viewer_cache.get("volume"))
        if "frames" in self._viewer_cache:
            try:
                setattr(app, "_viewer_frames", self._viewer_cache.get("frames"))
            except Exception:
                pass
        self._viewer_cache = {}
        if unlock_hook:
            self._unlock_and_disable_hook()
        self._request_render()

    def _reset_panel(self) -> None:
        self._restore_viewer(unlock_hook=True)
        self._maps = {}
        self._map_var.set("DWI")
        if self._map_combo is not None:
            self._map_combo["values"] = ["DWI"]
        self._info_var.set("")
        self._set_status("Ready.")

    def _resolve_scan_from_app(self) -> Any:
        scan = getattr(self._app, "_scan", None)
        if scan is not None:
            return scan
        try:
            state = getattr(self._app, "state", None)
            dataset = getattr(self._app, "dataset", None)
            selected = None
            if state is not None:
                selected = getattr(getattr(state, "dataset", None), "selected_scan_id", None)
            if selected is not None and dataset is not None:
                get_scan = getattr(dataset, "get_scan", None)
                if callable(get_scan):
                    return get_scan(int(selected))
        except Exception:
            return None
        return None

    def _resolve_viewer_volume(self) -> Optional[np.ndarray]:
        data = getattr(self._app, "_viewer_volume", None)
        if data is None:
            return None
        try:
            return np.asarray(data)
        except Exception:
            return None

    def _resolve_dwi_volume(self) -> Optional[np.ndarray]:
        if self._viewer_cache.get("volume") is not None:
            try:
                return np.asarray(self._viewer_cache["volume"])
            except Exception:
                return None
        return self._resolve_viewer_volume()

    def _resolve_viewer_affine(self) -> Optional[np.ndarray]:
        app = self._app
        if app is None:
            return None
        affine = getattr(app, "_viewer_raw_affine", None)
        if affine is None:
            return None
        if isinstance(affine, (list, tuple)):
            try:
                arr = np.asarray(affine, dtype=float)
            except Exception:
                arr = None
            if arr is not None:
                # (N,4,4) slice pack affines
                if arr.ndim == 3:
                    idx = 0
                    try:
                        state = getattr(app, "state", None)
                        viewer_state = getattr(state, "viewer", None) if state is not None else None
                        if viewer_state is not None:
                            idx = int(getattr(viewer_state, "slicepack_index", 0) or 0)
                    except Exception:
                        idx = 0
                    if idx < 0 or idx >= arr.shape[0]:
                        idx = 0
                    affine = arr[idx]
                # (4,4) single affine expressed as list of lists
                elif arr.ndim == 2:
                    affine = arr
        try:
            aff = np.asarray(affine, dtype=float)
        except Exception:
            return None

        raw = getattr(app, "_viewer_raw_volume", None)
        if raw is None:
            return aff
        try:
            from brkraw_viewer.utils.orientation import reorient_to_ras

            raw_arr = np.asarray(raw)
            ras_data, ras_affine = reorient_to_ras(raw_arr, aff)
            view_data = self._resolve_dwi_volume()
            if view_data is None:
                view_data = self._resolve_viewer_volume()
            if view_data is not None:
                if np.asarray(view_data).shape[:3] == ras_data.shape[:3]:
                    return np.asarray(ras_affine, dtype=float)
        except Exception:
            return aff
        return aff

    def _schedule_finalize(self) -> None:
        def update_ui() -> None:
            values = ["DWI", *self._maps.keys()]
            if self._map_combo is not None:
                self._map_combo["values"] = values
            self._map_var.set("DWI")
            self._restore_viewer()
            self._set_status("Done.")
            if self._calc_button is not None:
                self._calc_button.configure(state="normal")
            if self._save_button is not None:
                self._save_button.configure(state="normal" if self._maps else "disabled")
            try:
                messagebox.showinfo("DTI", "DTI calculation completed.")
            except Exception:
                pass

        self.after(0, update_ui)

    def _schedule_status(self, message: str) -> None:
        self.after(0, lambda: self._set_status(message))

    def _schedule_calc_ready(self) -> None:
        self.after(0, lambda: self._calc_button.configure(state="normal") if self._calc_button else None)

    def _set_status(self, message: str) -> None:
        self._status_var.set(message)

    def _slugify_metric(self, name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")
        return cleaned or "metric"

    def _resolve_output_prefix(self) -> Optional[Path]:
        app = self._app
        if app is None:
            return None
        try:
            output_dir = Path(getattr(app, "_convert_output_dir", Path.cwd()))
        except Exception:
            output_dir = Path.cwd()
        sid = getattr(getattr(app, "state", None), "dataset", None)
        scan_id = getattr(sid, "selected_scan_id", None)
        reco_id = getattr(sid, "selected_reco_id", None)
        if scan_id is None or reco_id is None:
            return None

        base_path: Optional[Path] = None
        planner = getattr(app, "_plan_output_paths", None)
        if callable(planner):
            planned = planner(
                output_dir=output_dir,
                base_name="",
                scan_id=int(scan_id),
                reco_id=int(reco_id),
                layout_source=getattr(app, "_convert_layout_source", None),
                layout_auto=getattr(app, "_convert_layout_auto", None),
                layout_template=getattr(app, "_convert_layout_template", None),
                context_map=getattr(app, "_context_map_path", None),
            )
            if isinstance(planned, (list, tuple)) and planned:
                base_path = Path(str(planned[0]))

        if base_path is None:
            base = f"scan{int(scan_id):03d}_reco{int(reco_id):03d}"
            base_path = output_dir / f"{base}.nii.gz"

        name = base_path.name
        if name.endswith(".nii.gz"):
            name = name[: -len(".nii.gz")]
        elif name.endswith(".nii"):
            name = name[: -len(".nii")]
        prefix = base_path.with_name(name)

        confirm = messagebox.askyesno(
            "DTI",
            f"Save DTI maps to:\n{prefix}_<metric>.nii.gz\n\nProceed?",
        )
        if not confirm:
            return None
        return prefix

    def _save_maps(self) -> None:
        if not self._maps:
            self._set_status("No DTI maps to save.")
            return
        prefix = self._resolve_output_prefix()
        if prefix is None:
            return
        affine = self._resolve_viewer_affine()
        if affine is None:
            affine = np.eye(4, dtype=float)
        errors: list[str] = []
        for name, data in self._maps.items():
            if data is None:
                continue
            try:
                arr = np.asarray(data)
                if arr.ndim == 0:
                    continue
                out_name = f"{prefix}_{self._slugify_metric(name)}.nii.gz"
                nii = Nifti1Image(arr.astype(np.float32, copy=False), affine)
                nii.to_filename(out_name)
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if errors:
            self._set_status("Some DTI maps failed to save.")
            try:
                messagebox.showwarning("DTI", "Save errors:\n" + "\n".join(errors))
            except Exception:
                pass
            return
        self._set_status("DTI maps saved.")

    def _lock_viewer_hook(self, locked: bool) -> None:
        app = self._app
        if app is None:
            return
        hook_name = getattr(app, "_viewer_hook_name", None)
        toggle = getattr(app, "on_viewer_hook_toggle", None)
        lock = getattr(app, "on_viewer_hook_lock", None)
        if callable(toggle):
            toggle(True, hook_name)
        if callable(lock):
            lock(bool(locked))

    def _unlock_and_disable_hook(self) -> None:
        app = self._app
        if app is None:
            return
        hook_name = getattr(app, "_viewer_hook_name", None)
        toggle = getattr(app, "on_viewer_hook_toggle", None)
        lock = getattr(app, "on_viewer_hook_lock", None)
        if callable(lock):
            lock(False)
        if callable(toggle):
            toggle(False, hook_name)

    def _request_render(self) -> None:
        renderer = getattr(self._app, "_render_viewer_views", None)
        if callable(renderer):
            try:
                renderer()
            except Exception:
                pass



def _crop_view(img: np.ndarray, *, center: tuple[int, int], zoom: float) -> np.ndarray:
    if zoom <= 1.0:
        return img
    h, w = img.shape[:2]
    crop_h = max(1, int(round(h / zoom)))
    crop_w = max(1, int(round(w / zoom)))
    cy, cx = center
    y0 = min(max(cy - crop_h // 2, 0), max(h - crop_h, 0))
    x0 = min(max(cx - crop_w // 2, 0), max(w - crop_w, 0))
    return img[y0 : y0 + crop_h, x0 : x0 + crop_w]


ViewerHook = DTIViewerHook
