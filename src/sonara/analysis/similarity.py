"""Cross-track similarity computation and distance metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from sonara.representation.schema import SMRDocument


@dataclass
class SimilarityMatch:
    """Pairwise similarity result."""
    track_a: str
    track_b: str
    cosine_similarity: float
    euclidean_distance: float


class SimilarityEngine:
    """Computes multidimensional similarity and distance between SMR documents."""

    @classmethod
    def extract_feature_vector(cls, doc: SMRDocument) -> np.ndarray:
        """
        Extract normalized numerical representation vector from an SMR document.

        Components:
            - Rhythmic (4 dims): BPM normalized, syncopation, pulse clarity, onset density.
            - Harmonic (13 dims): 12 chroma bins + tonal stability.
            - Timbral (6 dims): Centroid, bandwidth, warmth, brightness, flatness, dynamic range.
        Total: 23-dimensional feature embedding.
        """
        r = doc.rhythm
        h = doc.harmony
        a = doc.acoustic_features

        rhythm_vec = [
            min(1.0, (r.estimated_bpm or 120.0) / 240.0),
            r.syncopation_index,
            r.pulse_clarity,
            min(1.0, r.onset_density_per_second / 15.0),
        ]

        chroma_vec = h.chroma_profile if len(h.chroma_profile) == 12 else [1.0 / 12.0] * 12
        harmonic_vec = chroma_vec + [h.tonal_stability]

        timbral_vec = [
            min(1.0, a.spectral_centroid_mean / 5000.0),
            min(1.0, a.spectral_bandwidth_mean / 4000.0),
            a.warmth_index,
            a.brightness_index,
            a.spectral_flatness_mean,
            min(1.0, a.dynamic_range_db / 60.0),
        ]

        return np.array(rhythm_vec + harmonic_vec + timbral_vec, dtype=float)

    @classmethod
    def compute_pairwise(cls, doc_a: SMRDocument, doc_b: SMRDocument) -> SimilarityMatch:
        """Compute similarity metrics between two tracks."""
        vec_a = cls.extract_feature_vector(doc_a)
        vec_b = cls.extract_feature_vector(doc_b)

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a > 1e-8 and norm_b > 1e-8:
            cos_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        else:
            cos_sim = 0.0

        euc_dist = float(np.linalg.norm(vec_a - vec_b))

        return SimilarityMatch(
            track_a=doc_a.metadata.file_name,
            track_b=doc_b.metadata.file_name,
            cosine_similarity=round(float(np.clip(cos_sim, 0.0, 1.0)), 4),
            euclidean_distance=round(euc_dist, 4),
        )

    @classmethod
    def compute_distance_matrix(cls, documents: List[SMRDocument]) -> Tuple[List[str], np.ndarray]:
        """Compute complete NxN pairwise similarity matrix across a collection."""
        n = len(documents)
        names = [d.metadata.file_name for d in documents]
        vectors = np.array([cls.extract_feature_vector(d) for d in documents])

        # Normalize rows for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        norm_vectors = vectors / norms
        sim_matrix = np.dot(norm_vectors, norm_vectors.T)

        return names, np.clip(sim_matrix, 0.0, 1.0)
