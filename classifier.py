from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class RegionReaction:
    score: float
    positive: bool


def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid image data")
    if len(image_bgr.shape) == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _split_regions(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = gray.shape[:2]
    if w < 3:
        raise ValueError("Image is too small for 3 reagent regions")
    third = w // 3
    anti_a = gray[:, :third]
    anti_b = gray[:, third : 2 * third]
    anti_d = gray[:, 2 * third :]
    return anti_a, anti_b, anti_d


def _reaction_score(region: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(region, (5, 5), 0)
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    edge_energy = float(np.mean(np.abs(lap)))
    norm = min(edge_energy / 20.0, 1.0)
    return norm


def _reaction(region: np.ndarray, threshold: float) -> RegionReaction:
    score = _reaction_score(region)
    return RegionReaction(score=score, positive=score >= threshold)


def _group_from_reactions(a_pos: bool, b_pos: bool, d_pos: bool) -> str:
    if a_pos and b_pos:
        abo = "AB"
    elif a_pos:
        abo = "A"
    elif b_pos:
        abo = "B"
    else:
        abo = "O"
    rh = "+" if d_pos else "-"
    return f"{abo}{rh}"


def predict_blood_group(image_bgr: np.ndarray, threshold: float = 0.35) -> Dict[str, object]:
    gray = _to_gray(image_bgr)
    anti_a_r, anti_b_r, anti_d_r = _split_regions(gray)

    anti_a = _reaction(anti_a_r, threshold)
    anti_b = _reaction(anti_b_r, threshold)
    anti_d = _reaction(anti_d_r, threshold)

    blood_group = _group_from_reactions(anti_a.positive, anti_b.positive, anti_d.positive)

    scores = np.array([anti_a.score, anti_b.score, anti_d.score], dtype=float)
    separation_metric = float(np.clip(np.mean(np.abs(scores - threshold)) / max(threshold, 1e-6), 0.0, 1.0))

    return {
        "blood_group": blood_group,
        "reactions": {
            "anti_a": {"positive": anti_a.positive, "score": round(anti_a.score, 4)},
            "anti_b": {"positive": anti_b.positive, "score": round(anti_b.score, 4)},
            "anti_d": {"positive": anti_d.positive, "score": round(anti_d.score, 4)},
        },
        "confidence": round(separation_metric, 4),
    }
