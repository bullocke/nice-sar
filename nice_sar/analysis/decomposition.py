"""Polarimetric decomposition methods.

Implements Freeman-Durden and Cloude-Pottier decompositions for
quad-pol SAR covariance data. Includes vectorized implementations
for efficient processing of large arrays.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

from nice_sar._types import ArrayFloat32

logger = logging.getLogger(__name__)

# Pauli basis transformation matrix (lexicographic → coherency)
_PAULI_U = np.array(
    [
        [1 / np.sqrt(2), 0.0, 1 / np.sqrt(2)],
        [1 / np.sqrt(2), 0.0, -1 / np.sqrt(2)],
        [0.0, np.sqrt(2.0), 0.0],
    ],
    dtype=np.complex64,
)


def build_coherency_matrix(
    covariances: dict[str, np.ndarray],
    window: int = 7,
) -> np.ndarray:
    """Build and spatially average the 3x3 coherency matrix T from covariance elements.

    Args:
        covariances: Dictionary from :func:`nice_sar.io.products.read_quad_covariances`
            with keys ``HHHH``, ``HVHV``, ``VVVV``, ``HHHV``, ``HHVV``, ``HVVV``.
        window: Spatial averaging window size for ensemble estimation.

    Returns:
        Array of shape ``(3, 3, H, W)`` containing the coherency matrix at each pixel.
    """

    def _smooth(arr: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(arr):
            mr = ndimage.uniform_filter(arr.real, size=window, mode="nearest")
            mi = ndimage.uniform_filter(arr.imag, size=window, mode="nearest")
            return mr + 1j * mi
        return ndimage.uniform_filter(arr, size=window, mode="nearest")

    c11 = _smooth(covariances["HHHH"])
    c22 = _smooth(covariances["HVHV"])
    c33 = _smooth(covariances["VVVV"])
    c12 = _smooth(covariances["HHHV"])
    c13 = _smooth(covariances["HHVV"])
    c23 = _smooth(covariances["HVVV"])

    rows, cols = c11.shape

    # Build covariance matrix C: (3, 3, H, W)
    C = np.zeros((3, 3, rows, cols), dtype=np.complex64)
    C[0, 0] = c11
    C[0, 1] = c12
    C[0, 2] = c13
    C[1, 0] = np.conj(c12)
    C[1, 1] = c22
    C[1, 2] = c23
    C[2, 0] = np.conj(c13)
    C[2, 1] = np.conj(c23)
    C[2, 2] = c33

    # Transform to coherency: T = U @ C @ U^H (vectorized over pixels)
    # C is (3,3,H,W) → reshape to (3,3,N) for batched matmul
    N = rows * cols
    C_flat = C.reshape(3, 3, N)

    # Use einsum for batched matrix multiplication: T = U @ C @ U^H
    UC = np.einsum("ij,jkn->ikn", _PAULI_U, C_flat)
    T_flat = np.einsum("ijn,kj->ikn", UC, np.conj(_PAULI_U))

    return T_flat.reshape(3, 3, rows, cols)


def freeman_durden(T: np.ndarray) -> tuple[ArrayFloat32, ArrayFloat32, ArrayFloat32]:
    """Freeman-Durden 3-component decomposition (vectorized).

    Decomposes the coherency matrix into surface (Ps), double-bounce (Pd),
    and volume (Pv) scattering power components.

    Args:
        T: Coherency matrix array of shape ``(3, 3, H, W)``.

    Returns:
        Tuple of ``(Ps, Pd, Pv)`` arrays, each of shape ``(H, W)``.
    """
    T11 = T[0, 0].real
    T22 = T[1, 1].real
    T33 = T[2, 2].real
    T13 = np.abs(T[0, 2]).real

    Pv = 2.0 * T33
    Pd = np.maximum(T22 - (T33 - T13), 0.0)
    Ps = np.maximum(T11 - (T33 - T13), 0.0)

    return Ps.astype(np.float32), Pd.astype(np.float32), Pv.astype(np.float32)


def cloude_pottier(
    T: np.ndarray,
) -> tuple[ArrayFloat32, ArrayFloat32, ArrayFloat32]:
    """Cloude-Pottier H/A/alpha decomposition (vectorized).

    Computes entropy (H), anisotropy (A), and mean alpha angle from the
    eigenvalue decomposition of the coherency matrix.

    Args:
        T: Coherency matrix array of shape ``(3, 3, H, W)``.

    Returns:
        Tuple of ``(H, A, alpha)`` where H ∈ [0,1], A ∈ [0,1],
        alpha ∈ [0, 90] degrees.
    """
    rows, cols = T.shape[2], T.shape[3]
    N = rows * cols

    # Reshape to (N, 3, 3) for batch eigendecomposition
    T_batch = T.reshape(3, 3, N).transpose(2, 0, 1)

    # Hermitian symmetrize
    T_batch = (T_batch + np.conj(T_batch.transpose(0, 2, 1))) / 2.0

    H_out = np.full(N, np.nan, dtype=np.float32)
    A_out = np.full(N, np.nan, dtype=np.float32)
    alpha_out = np.full(N, np.nan, dtype=np.float32)

    # Process in chunks to manage memory
    chunk_size = 10000
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = T_batch[start:end]

        # Skip pixels with NaN
        valid_mask = np.all(np.isfinite(chunk.reshape(end - start, -1)), axis=1)
        if not np.any(valid_mask):
            continue

        valid_chunk = chunk[valid_mask]
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(valid_chunk)
        except np.linalg.LinAlgError:
            continue

        # Sort eigenvalues descending
        idx = np.argsort(eigenvalues, axis=1)[:, ::-1]
        w = np.take_along_axis(eigenvalues, idx, axis=1).real
        v = np.take_along_axis(
            eigenvectors, idx[:, np.newaxis, :].repeat(3, axis=1), axis=2
        )

        w = np.maximum(w, 0.0)
        total = np.sum(w, axis=1, keepdims=True) + 1e-12
        ps = w / total

        # Entropy
        h = -np.sum(ps * np.log(ps + 1e-12), axis=1) / np.log(3.0)

        # Anisotropy
        a = (ps[:, 1] - ps[:, 2]) / (ps[:, 1] + ps[:, 2] + 1e-12)

        # Alpha from dominant eigenvector
        v1_abs2 = np.abs(v[:, 0, 0]) ** 2
        v2_abs2 = np.abs(v[:, 1, 0]) ** 2
        v3_abs2 = np.abs(v[:, 2, 0]) ** 2 + 1e-12
        al = np.degrees(np.arctan(np.sqrt((v2_abs2 + v1_abs2) / (2.0 * v3_abs2))))

        # Write back
        valid_indices = np.arange(start, end)[valid_mask]
        H_out[valid_indices] = np.clip(h, 0, 1)
        A_out[valid_indices] = np.clip(a, 0, 1)
        alpha_out[valid_indices] = np.clip(al, 0, 90)

    return (
        H_out.reshape(rows, cols),
        A_out.reshape(rows, cols),
        alpha_out.reshape(rows, cols),
    )


def compute_pauli_rgb(
    hh: np.ndarray,
    hv: np.ndarray,
    vv: np.ndarray,
) -> np.ndarray | None:
    """Compute Pauli RGB composite from quad-pol data.

    Assigns: R = |HH - VV|, G = sqrt(2) * |HV|, B = |HH + VV|.

    Args:
        hh: HH backscatter (linear power).
        hv: HV backscatter (linear power).
        vv: VV backscatter (linear power).

    Returns:
        Array of shape ``(3, H, W)`` in float32, or ``None`` if inputs are None.
    """
    R = np.abs(hh - vv).astype(np.float32)
    G = (np.sqrt(2.0) * np.abs(hv)).astype(np.float32)
    B = np.abs(hh + vv).astype(np.float32)
    return np.stack([R, G, B], axis=0)
