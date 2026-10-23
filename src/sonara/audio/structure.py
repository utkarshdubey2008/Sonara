"""Musical structure segmentation, recurrence analysis, and section labeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class SectionBoundary:
    """Detected structural segment."""
    section_id: str
    section_type: str  # intro, verse, chorus, bridge, outro, transition, body
    start_time: float
    end_time: float
    duration: float
    confidence: float
    is_repetition: bool = False
    repeat_of_id: Optional[str] = None


@dataclass
class StructuralAnalysisResult:
    """Results from structure and recurrence analysis."""
    sections: List[SectionBoundary]
    num_sections: int
    repetition_density: float
    average_section_duration: float
    uncertainty_flags: List[str] = field(default_factory=list)


class StructureAnalyzer:
    """Identifies structural transitions and recurrent sections in audio."""

    def __init__(self, hop_length: int = 1024, min_segment_duration: float = 3.0):
        self.hop_length = hop_length
        self.min_segment_duration = min_segment_duration

    def analyze(self, y: np.ndarray, sr: int) -> StructuralAnalysisResult:
        """
        Segment audio into musical sections using self-similarity and novelty filtering.

        Args:
            y: Audio waveform array.
            sr: Sample rate.

        Returns:
            StructuralAnalysisResult instance.
        """
        uncertainty_flags: List[str] = []
        duration = float(len(y)) / float(sr) if sr > 0 else 0.0

        if duration < self.min_segment_duration * 1.5:
            # Too short for multi-section segmentation
            sec = SectionBoundary(
                section_id="sec_0",
                section_type="body",
                start_time=0.0,
                end_time=round(duration, 3),
                duration=round(duration, 3),
                confidence=1.0,
                is_repetition=False,
            )
            return StructuralAnalysisResult(
                sections=[sec],
                num_sections=1,
                repetition_density=0.0,
                average_section_duration=round(duration, 3),
                uncertainty_flags=["track_too_short_for_segmentation"],
            )

        # 1. Extract combined feature representation (Chroma + MFCC)
        try:
            chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=self.hop_length)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            # Normalize and stack
            mfcc_norm = mfcc / (np.linalg.norm(mfcc, axis=0, keepdims=True) + 1e-6)
            features = np.vstack([chroma, mfcc_norm])
        except Exception as e:
            uncertainty_flags.append(f"feature_extraction_failed: {str(e)}")
            sec = SectionBoundary(
                section_id="sec_0",
                section_type="body",
                start_time=0.0,
                end_time=round(duration, 3),
                duration=round(duration, 3),
                confidence=0.5,
            )
            return StructuralAnalysisResult(
                sections=[sec],
                num_sections=1,
                repetition_density=0.0,
                average_section_duration=round(duration, 3),
                uncertainty_flags=uncertainty_flags,
            )

        n_frames = features.shape[1]
        frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=self.hop_length)

        # 2. Compute Self-Similarity Matrix (SSM)
        # Cosine affinity
        features_unit = features / (np.linalg.norm(features, axis=0, keepdims=True) + 1e-8)
        ssm = np.dot(features_unit.T, features_unit)

        # 3. Novelty curve via checkerboard kernel convolution
        kernel_size = min(32, max(8, n_frames // 16))
        novelty_curve = self._compute_novelty_curve(ssm, kernel_size)

        # 4. Peak picking for boundaries
        min_frames = int(self.min_segment_duration * sr / self.hop_length)
        boundary_frames = [0]

        # Find local peaks exceeding threshold
        threshold = np.mean(novelty_curve) + 0.5 * np.std(novelty_curve)
        for i in range(kernel_size, n_frames - kernel_size):
            if novelty_curve[i] > threshold:
                if (
                    novelty_curve[i] == np.max(novelty_curve[max(0, i - min_frames // 2) : min(n_frames, i + min_frames // 2 + 1)])
                    and (i - boundary_frames[-1]) >= min_frames
                ):
                    boundary_frames.append(i)

        if (n_frames - boundary_frames[-1]) >= min_frames // 2:
            boundary_frames.append(n_frames - 1)
        else:
            boundary_frames[-1] = n_frames - 1

        # 5. Build section segments
        raw_sections: List[Tuple[float, float, np.ndarray]] = []
        for i in range(len(boundary_frames) - 1):
            f_start = boundary_frames[i]
            f_end = boundary_frames[i + 1]
            t_start = float(frame_times[f_start])
            t_end = float(frame_times[f_end])
            seg_feat = np.mean(features[:, f_start:f_end], axis=1)
            raw_sections.append((t_start, t_end, seg_feat))

        # 6. Cluster / match sections for repetition and labeling
        sections = self._label_sections(raw_sections, duration)

        reps = sum(1 for s in sections if s.is_repetition)
        rep_density = float(reps / len(sections)) if sections else 0.0
        avg_dur = float(np.mean([s.duration for s in sections])) if sections else duration

        return StructuralAnalysisResult(
            sections=sections,
            num_sections=len(sections),
            repetition_density=round(rep_density, 3),
            average_section_duration=round(avg_dur, 2),
            uncertainty_flags=uncertainty_flags,
        )

    def _compute_novelty_curve(self, ssm: np.ndarray, k: int) -> np.ndarray:
        """Convolve SSM diagonal with a 2k x 2k Foote checkerboard kernel."""
        n = ssm.shape[0]
        # Foote checkerboard kernel: +1 along diagonal quadrants, -1 on off-diagonal quadrants
        kernel = np.kron([[1, -1], [-1, 1]], np.ones((k, k)))
        window = np.outer(np.hanning(2 * k), np.hanning(2 * k))
        tapered_kernel = kernel * window

        novelty = np.zeros(n)
        for i in range(k, n - k):
            patch = ssm[i - k : i + k, i - k : i + k]
            novelty[i] = np.sum(patch * tapered_kernel)

        # Half-wave rectification and smoothing
        novelty = np.maximum(0, novelty)
        novelty = gaussian_filter1d(novelty, sigma=2.0)
        norm = np.max(novelty)
        if norm > 1e-6:
            novelty /= norm
        return novelty

    def _label_sections(
        self, raw_sections: List[Tuple[float, float, np.ndarray]], total_duration: float
    ) -> List[SectionBoundary]:
        """Assign musical labels (intro, verse, chorus, outro) based on position and similarity."""
        n = len(raw_sections)
        if n == 0:
            return []

        # Pairwise similarity matrix between segments
        sim_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                vi = raw_sections[i][2]
                vj = raw_sections[j][2]
                dot = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-8)
                sim_mat[i, j] = dot

        results: List[SectionBoundary] = []
        label_counts: Dict[str, int] = {"verse": 0, "chorus": 0}

        for i, (t_start, t_end, _) in enumerate(raw_sections):
            sec_id = f"sec_{i}"
            seg_dur = t_end - t_start

            # Find if this matches any previous section
            repeat_of: Optional[str] = None
            is_rep = False
            assigned_label = "body"

            for prev_idx in range(i):
                if sim_mat[i, prev_idx] >= 0.82:  # High acoustic/harmonic similarity threshold
                    is_rep = True
                    repeat_of = f"sec_{prev_idx}"
                    assigned_label = results[prev_idx].section_type
                    break

            if not is_rep:
                # Heuristic positional assignment
                if i == 0 and t_start < 2.0 and seg_dur <= total_duration * 0.25:
                    assigned_label = "intro"
                elif i == n - 1 and (total_duration - t_end) < 2.0:
                    assigned_label = "outro"
                else:
                    if label_counts["verse"] <= label_counts["chorus"]:
                        assigned_label = "verse"
                        label_counts["verse"] += 1
                    else:
                        assigned_label = "chorus"
                        label_counts["chorus"] += 1

            results.append(
                SectionBoundary(
                    section_id=sec_id,
                    section_type=assigned_label,
                    start_time=round(t_start, 3),
                    end_time=round(t_end, 3),
                    duration=round(seg_dur, 3),
                    confidence=0.85 if is_rep else 0.70,
                    is_repetition=is_rep,
                    repeat_of_id=repeat_of,
                )
            )

        return results
