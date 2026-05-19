"""
2D growth-forecast visualisation: an MRI slice with the current tumor
segmentation and per-timepoint overlays showing where the tumor is
predicted to be at each future timepoint.

The model predicts volumes (scalars), not future segmentation masks. So the
spatial overlay here is an *isotropic-growth approximation*: we expand or
contract the current 2D mask uniformly so its area scales as
(V_future / V_current) ** (2/3) — the area-vs-volume law you'd get if the
tumor grew uniformly outward in 3D. This is a deliberately simple model.

Two rendering helpers are exposed:

  * `render_growth_view`   — single panel, MRI background, baseline filled,
                             growth/shrinkage filled in distinct colours,
                             contour outline per FU, optional tumor crop.
  * `render_side_by_side`  — one panel per timepoint, each showing the
                             baseline outline + that single timepoint's
                             extent. Useful for clean reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class GrowthLayer:
    """A predicted future tumor extent in a single 2D slice."""
    label: str            # "FU3", "FU4", "FU5"
    target_area_px: int
    mask_2d: np.ndarray   # bool, same shape as the slice
    volume_mm3: float


# ---------------------------------------------------------------- slice picking

def find_largest_tumor_slice(mask_3d: np.ndarray) -> int:
    """Return the axial-slice index (last axis) with the most tumor pixels.
    Raises ValueError if the mask is empty."""
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask_3d.shape}")
    binary = mask_3d > 0
    if not binary.any():
        raise ValueError("Mask contains no tumor voxels")
    return int(np.argmax(binary.sum(axis=(0, 1))))


# ---------------------------------------------------------------- area dilation

def grow_mask_2d_to_area(mask_2d: np.ndarray, target_area_px: int) -> np.ndarray:
    """
    Return a boolean 2D mask whose area in pixels is as close as possible to
    `target_area_px`, obtained by uniform inward/outward growth from the
    boundary of the input mask.

    Implementation: distance transform from the boundary, then threshold at
    the level set that yields the desired area.
    """
    if mask_2d.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask_2d.shape}")
    mask = mask_2d.astype(bool)
    current_area = int(mask.sum())
    h, w = mask.shape
    canvas_area = h * w
    target_area_px = max(0, min(int(target_area_px), canvas_area))

    if target_area_px == current_area:
        return mask.copy()

    if target_area_px > current_area:
        if not mask.any():
            return mask.copy()
        # Distance from each outside pixel to the nearest tumor pixel.
        # Ties are broken stably so we always emit *exactly* the requested
        # area rather than over-shooting on tied distances.
        dist_outside = distance_transform_edt(~mask)
        outside_rows, outside_cols = np.where(~mask)
        outside_dists = dist_outside[outside_rows, outside_cols]
        needed = target_area_px - current_area
        if needed >= outside_dists.size:
            return np.ones_like(mask)
        order = np.argsort(outside_dists, kind="stable")[:needed]
        out = mask.copy()
        out[outside_rows[order], outside_cols[order]] = True
        return out

    # target < current → erode by removing the outermost pixels
    if not mask.any():
        return mask.copy()
    keep = int(target_area_px)
    if keep == 0:
        return np.zeros_like(mask)
    dist_inside = distance_transform_edt(mask)
    inside_rows, inside_cols = np.where(mask)
    inside_dists = dist_inside[inside_rows, inside_cols]
    # Keep the `keep` interior pixels furthest from the boundary; ties broken stably.
    order = np.argsort(-inside_dists, kind="stable")[:keep]
    out = np.zeros_like(mask)
    out[inside_rows[order], inside_cols[order]] = True
    return out


# ---------------------------------------------------------------- main API

def predicted_growth_layers_2d(
    seg_3d: np.ndarray,
    voxel_dims: Tuple[float, float, float],
    current_volume_mm3: float,
    predicted_volumes_mm3: List[float],
    timepoint_labels: List[str],
    slice_index: Optional[int] = None,
) -> Tuple[int, np.ndarray, List[GrowthLayer]]:
    """
    Compute the 2D current mask + future-extent masks for a single axial slice.

    Returns
    -------
    slice_index : int
        The axial slice that was rendered (the largest-tumor slice unless
        explicitly given).
    current_mask_2d : np.ndarray
        Bool 2D mask of the current segmentation at that slice.
    layers : list[GrowthLayer]
        One layer per future timepoint, ordered as `timepoint_labels`.
    """
    if len(predicted_volumes_mm3) != len(timepoint_labels):
        raise ValueError("predicted_volumes and labels must be the same length")
    if current_volume_mm3 <= 0:
        raise ValueError("current_volume_mm3 must be > 0")
    if seg_3d.ndim != 3:
        raise ValueError("seg_3d must be a 3D array")

    if slice_index is None:
        slice_index = find_largest_tumor_slice(seg_3d)

    current_mask_2d = (seg_3d[:, :, slice_index] > 0)
    current_area_px = int(current_mask_2d.sum())

    layers: List[GrowthLayer] = []
    for label, v_future in zip(timepoint_labels, predicted_volumes_mm3):
        # Isotropic-growth area scaling: A_future / A_current = (V_future / V_current)^(2/3)
        ratio = max(float(v_future) / float(current_volume_mm3), 0.0)
        area_scale = ratio ** (2.0 / 3.0)
        target_area = int(round(current_area_px * area_scale))
        future_mask = grow_mask_2d_to_area(current_mask_2d, target_area)
        layers.append(GrowthLayer(
            label=label,
            target_area_px=target_area,
            mask_2d=future_mask,
            volume_mm3=float(v_future),
        ))

    return slice_index, current_mask_2d, layers


def stacked_timeline_heatmap(
    current_mask_2d: np.ndarray,
    layers: List[GrowthLayer],
) -> np.ndarray:
    """
    Build a single 2D array suitable for `imshow` where each pixel encodes
    the *earliest predicted timepoint at which it is tumor*.

    Encoding (float-valued so colormaps interpolate smoothly):
        NaN  → never tumor in the forecast horizon
        0.0  → already tumor at baseline / current
        i    → first becomes tumor at layer i (1, 2, 3, …)

    Layers must be passed in time order (FU3, FU4, FU5).
    """
    out = np.full(current_mask_2d.shape, np.nan, dtype=float)
    out[current_mask_2d] = 0.0
    for i, layer in enumerate(layers, start=1):
        new_pixels = layer.mask_2d & np.isnan(out)
        out[new_pixels] = float(i)
    return out


# ---------------------------------------------------------------- demo fallback

def synthetic_disk_mask(side: int = 200, radius: int = 25,
                        center: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """A circular binary mask used for the demo-cohort illustration mode."""
    if center is None:
        center = (side // 2, side // 2)
    yy, xx = np.ogrid[:side, :side]
    return (((xx - center[1]) ** 2 + (yy - center[0]) ** 2) <= radius ** 2)


# ---------------------------------------------------------------- rendering

# Per-future-timepoint contour styling
_FU_COLORS  = ["#ffb300", "#ff7043", "#e53935"]   # FU3, FU4, FU5
_FU_STYLES  = ["--", "-.", ":"]


def _bbox_with_margin(masks: List[np.ndarray], margin_frac: float = 0.4,
                      min_margin_px: int = 12) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box around the union of given masks, expanded by a margin."""
    if not masks:
        return None
    union = masks[0].copy()
    for m in masks[1:]:
        union = union | m
    if not union.any():
        return None
    rows = np.any(union, axis=1)
    cols = np.any(union, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    height = rmax - rmin + 1
    width  = cmax - cmin + 1
    margin = max(min_margin_px, int(margin_frac * max(height, width)))
    rmin = max(0, rmin - margin)
    cmin = max(0, cmin - margin)
    rmax = min(union.shape[0] - 1, rmax + margin)
    cmax = min(union.shape[1] - 1, cmax + margin)
    return int(rmin), int(rmax), int(cmin), int(cmax)


def render_growth_view(
    ax,
    background: np.ndarray,
    current_2d: np.ndarray,
    layers: List[GrowthLayer],
    current_volume_mm3: float,
    crop_to_tumor: bool = True,
):
    """
    Single panel: MRI background, current tumor filled red, growth
    filled yellow, shrinkage filled cyan, plus a dashed contour per FU.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    bg = background
    cur = current_2d
    layer_masks = [L.mask_2d for L in layers]

    if crop_to_tumor:
        bbox = _bbox_with_margin([cur, *layer_masks])
        if bbox is not None:
            r0, r1, c0, c1 = bbox
            bg = bg[r0:r1 + 1, c0:c1 + 1]
            cur = cur[r0:r1 + 1, c0:c1 + 1]
            layer_masks = [m[r0:r1 + 1, c0:c1 + 1] for m in layer_masks]

    if bg is not None and bg.size:
        p1, p99 = np.percentile(bg, [1, 99]) if bg.dtype != bool else (0, 1)
        ax.imshow(np.clip(bg, p1, p99) if p99 > p1 else bg, cmap="gray")
    else:
        ax.imshow(np.zeros_like(cur), cmap="gray", vmin=0, vmax=1)

    # Final-FU envelope drives growth/shrink fills (worst-case extent)
    final_mask = layer_masks[-1] if layer_masks else cur
    growth = final_mask & ~cur
    shrink = cur & ~final_mask

    h, w = cur.shape
    if cur.any():
        red = np.zeros((h, w, 4))
        red[cur] = [1.0, 0.20, 0.20, 0.55]
        ax.imshow(red)
    if growth.any():
        yellow = np.zeros((h, w, 4))
        yellow[growth] = [1.0, 0.90, 0.10, 0.45]
        ax.imshow(yellow)
    if shrink.any():
        cyan = np.zeros((h, w, 4))
        cyan[shrink] = [0.20, 0.75, 1.00, 0.45]
        ax.imshow(cyan)

    # Solid red current outline + dashed contours per FU
    if cur.any():
        ax.contour(cur, levels=[0.5], colors="red", linewidths=2.0)
    for layer, m, color, style in zip(layers, layer_masks, _FU_COLORS, _FU_STYLES):
        if m.any():
            ax.contour(m, levels=[0.5], colors=color, linewidths=1.5, linestyles=style)

    # Legend
    handles: List = []
    handles.append(Patch(facecolor=(1, 0.20, 0.20, 0.7),
                         label=f"Baseline · {current_volume_mm3:.0f} mm³"))
    if growth.any():
        handles.append(Patch(facecolor=(1, 0.90, 0.10, 0.7),
                             label="Predicted growth (vs FU5)"))
    if shrink.any():
        handles.append(Patch(facecolor=(0.20, 0.75, 1, 0.7),
                             label="Predicted shrinkage (vs FU5)"))
    for layer, color, style in zip(layers, _FU_COLORS, _FU_STYLES):
        delta = layer.volume_mm3 - current_volume_mm3
        sign = "+" if delta >= 0 else "−"
        handles.append(Line2D([0], [0], color=color, linestyle=style, linewidth=2,
                              label=f"{layer.label} · {layer.volume_mm3:.0f} mm³  ({sign}{abs(delta):.0f})"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def render_side_by_side(
    fig,
    background: np.ndarray,
    current_2d: np.ndarray,
    layers: List[GrowthLayer],
    current_volume_mm3: float,
    crop_to_tumor: bool = True,
):
    """
    Multi-panel: one axes per timepoint (Baseline + each FU). Each panel
    shows the same MRI background; the current tumor outline is drawn in
    every panel for reference, and that panel's predicted extent is filled.
    """
    layer_masks = [L.mask_2d for L in layers]
    bbox = _bbox_with_margin([current_2d, *layer_masks]) if crop_to_tumor else None
    if bbox is not None:
        r0, r1, c0, c1 = bbox
        bg_c = background[r0:r1 + 1, c0:c1 + 1]
        cur_c = current_2d[r0:r1 + 1, c0:c1 + 1]
        layer_masks_c = [m[r0:r1 + 1, c0:c1 + 1] for m in layer_masks]
    else:
        bg_c, cur_c, layer_masks_c = background, current_2d, layer_masks

    n = 1 + len(layers)
    axes = fig.subplots(1, n)
    if n == 1:
        axes = [axes]
    p1, p99 = np.percentile(bg_c, [1, 99]) if bg_c.size else (0, 1)

    # Panel 0: Baseline alone
    ax = axes[0]
    ax.imshow(np.clip(bg_c, p1, p99) if p99 > p1 else bg_c, cmap="gray")
    if cur_c.any():
        red = np.zeros((*cur_c.shape, 4)); red[cur_c] = [1, 0.2, 0.2, 0.55]
        ax.imshow(red)
        ax.contour(cur_c, levels=[0.5], colors="red", linewidths=2.0)
    ax.set_title(f"Baseline\n{current_volume_mm3:.0f} mm³", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    for i, (layer, m, color) in enumerate(zip(layers, layer_masks_c, _FU_COLORS), start=1):
        ax = axes[i]
        ax.imshow(np.clip(bg_c, p1, p99) if p99 > p1 else bg_c, cmap="gray")
        # Reference: faint baseline outline
        if cur_c.any():
            ax.contour(cur_c, levels=[0.5], colors="red", linewidths=1.0, alpha=0.7)
        # This timepoint's extent filled in its FU colour
        if m.any():
            rgba = np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]) / 255.0
            fill = np.zeros((*m.shape, 4))
            fill[m] = [rgba[0], rgba[1], rgba[2], 0.5]
            ax.imshow(fill)
            ax.contour(m, levels=[0.5], colors=color, linewidths=1.5)
        delta = layer.volume_mm3 - current_volume_mm3
        sign = "+" if delta >= 0 else "−"
        ax.set_title(f"{layer.label}\n{layer.volume_mm3:.0f} mm³  ({sign}{abs(delta):.0f})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    return axes
