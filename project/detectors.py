"""
Probe detectors for activations.
"""

import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from project.activations import Activations
from project.scores import Scores
from project.types import Label


class Detector(ABC):
    """Base class for detectors."""

    @abstractmethod
    def fit(
        self,
        honest_activations: Activations,
        deceptive_activations: Activations,
    ) -> None:
        """Fits the detector."""
        ...

    @abstractmethod
    def score(self, activations: Activations) -> Scores:
        """Scores activations."""
        ...

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "Detector":
        ...


class LogisticRegressionDetector(Detector):
    """Logistic regression probe on masked token activations."""

    def __init__(
        self,
        layers: List[int],
        reg_coeff: float,
        *,
        scale: bool = True,
        train_data: Optional[str] = None,
    ):
        self.layers = layers
        self.reg_coeff = reg_coeff
        self.scale = scale
        self.train_data = train_data
        self.clf: Optional[LogisticRegression] = None
        self.scaler_mean: Optional[Tensor] = None
        self.scaler_scale: Optional[Tensor] = None
        self.directions: Optional[Tensor] = None
        self.val_auroc: Optional[float] = None
        self.val_margin: Optional[float] = None
        self.val_n_total: Optional[int] = None
        self.val_n_honest: Optional[int] = None
        self.val_n_deceptive: Optional[int] = None

    def _get_features(self, activations: Activations) -> np.ndarray:
        """Returns token-level features."""
        masked = activations.get_masked_activations()
        n_tok, n_layers, dim = masked.shape
        return masked.cpu().float().numpy().reshape(n_tok, n_layers * dim)

    @staticmethod
    def _dialogue_counts(activations: Activations) -> Optional[list]:
        counts = getattr(activations, "detect_token_count_per_dialogue", None)
        return None if counts is None else [int(c) for c in counts]

    @staticmethod
    def _pool_by_dialogue(X_tok: np.ndarray, counts: list) -> np.ndarray:
        if sum(counts) != len(X_tok):
            raise ValueError(f"token/count mismatch: {sum(counts)} vs {len(X_tok)}")
        out, i = [], 0
        for c in counts:
            out.append(X_tok[i:i + c].mean(axis=0))
            i += c
        return np.stack(out)

    def fit(
        self,
        honest_activations: Activations,
        deceptive_activations: Activations,
        *,
        val_fraction: float = 0.0,
        val_seed: int = 42,
        shuffle_labels: bool = False,
        shuffle_seed: int = 0,
        unit: str = "dialogue",
        groups_honest: Optional[list] = None,
        groups_deceptive: Optional[list] = None,
        cv_folds: int = 0,
    ) -> None:
        """Fit LR probe.
        """
        X_h_tok = self._get_features(honest_activations)
        X_d_tok = self._get_features(deceptive_activations)
        counts_h = self._dialogue_counts(honest_activations)
        counts_d = self._dialogue_counts(deceptive_activations)
        if counts_h is None or counts_d is None:
            if unit == "dialogue" or (val_fraction and val_fraction > 0.0):
                raise ValueError(
                    "fit requires detect_token_count_per_dialogue on both activation sets "
                    "(needed for dialogue pooling and grouped splits); pass unit='token' with "
                    "val_fraction=0 only for legacy shims."
                )
            counts_h = [1] * len(X_h_tok)
            counts_d = [1] * len(X_d_tok)

        g_h = list(groups_honest) if groups_honest is not None else list(range(len(counts_h)))
        g_d = list(groups_deceptive) if groups_deceptive is not None else list(range(len(counts_d)))
        if groups_honest is None and len(counts_h) != len(counts_d):
            raise ValueError(
                "honest/deceptive dialogue counts differ and no explicit groups were given: "
                "default index pairing cannot guarantee fact-pair integrity. Pass "
                "groups_honest/groups_deceptive explicitly."
            )

        if unit == "dialogue":
            X = np.vstack([self._pool_by_dialogue(X_h_tok, counts_h),
                           self._pool_by_dialogue(X_d_tok, counts_d)])
            y = np.array([0] * len(counts_h) + [1] * len(counts_d))
            groups = np.array(g_h + g_d)
            sample_weight = None
        elif unit == "token":
            X = np.vstack([X_h_tok, X_d_tok])
            y = np.array([0] * len(X_h_tok) + [1] * len(X_d_tok))
            groups = np.array([g for g, c in zip(g_h, counts_h) for _ in range(c)]
                              + [g for g, c in zip(g_d, counts_d) for _ in range(c)])
            sample_weight = np.array([1.0 / c for c in counts_h for _ in range(c)]
                                     + [1.0 / c for c in counts_d for _ in range(c)])
        else:
            raise ValueError(f"unknown unit {unit!r} (expected 'dialogue' or 'token')")
        self.fit_unit = unit

        if shuffle_labels:
            y = np.random.default_rng(int(shuffle_seed)).permutation(y)

        _, n_flat = X.shape
        n_layers = len(self.layers)
        n_features = n_flat // n_layers
        if n_layers * n_features != n_flat:
            raise ValueError(
                f"Feature shape mismatch: n_flat={n_flat}, n_layers={n_layers}, n_features={n_features}"
            )

        if cv_folds and cv_folds > 1:
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import GroupKFold

            fold_aurocs, fold_margins = [], []
            for tr_idx, va_idx in GroupKFold(n_splits=int(cv_folds)).split(X, y, groups=groups):
                if len(np.unique(y[va_idx])) < 2:
                    continue
                if self.scale:
                    sc = StandardScaler().fit(X[tr_idx])
                    Xt, Xv = sc.transform(X[tr_idx]), sc.transform(X[va_idx])
                else:
                    Xt, Xv = X[tr_idx], X[va_idx]
                clf = LogisticRegression(C=1.0 / self.reg_coeff, max_iter=1000,
                                         random_state=42, fit_intercept=False)
                sw = sample_weight[tr_idx] if sample_weight is not None else None
                clf.fit(Xt, y[tr_idx], sample_weight=sw)
                s = clf.decision_function(Xv)
                fold_aurocs.append(float(roc_auc_score(y[va_idx], s)))
                fold_margins.append(float(s[y[va_idx] == 1].mean() - s[y[va_idx] == 0].mean()))
            self.val_auroc = float(np.mean(fold_aurocs)) if fold_aurocs else None
            self.val_auroc_sd = float(np.std(fold_aurocs, ddof=1)) if len(fold_aurocs) > 1 else 0.0
            self.val_margin = float(np.mean(fold_margins)) if fold_margins else None
            self.val_n_total = int(len(y))
            self.val_n_honest = int((y == 0).sum())
            self.val_n_deceptive = int((y == 1).sum())
            self.val_cv_folds = int(cv_folds)
            X_val = y_val = None
            X_train, y_train, sw_train = X, y, sample_weight
        else:
            X_val = y_val = None
            sw_train = sample_weight
            if val_fraction and val_fraction > 0.0:
                from sklearn.model_selection import GroupShuffleSplit
                gss = GroupShuffleSplit(n_splits=1, test_size=float(val_fraction),
                                        random_state=int(val_seed))
                tr_idx, va_idx = next(gss.split(X, y, groups=groups))
                X_train, X_val = X[tr_idx], X[va_idx]
                y_train, y_val = y[tr_idx], y[va_idx]
                if sample_weight is not None:
                    sw_train = sample_weight[tr_idx]
            else:
                X_train, y_train = X, y

        if self.scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scaler_mean = torch.tensor(scaler.mean_.reshape(n_layers, n_features), dtype=torch.float32)
            self.scaler_scale = torch.tensor(scaler.scale_.reshape(n_layers, n_features), dtype=torch.float32)
            X_val_scaled = (X_val - scaler.mean_) / scaler.scale_ if X_val is not None else None
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val

        self.clf = LogisticRegression(
            C=1.0 / self.reg_coeff,
            max_iter=1000,
            random_state=42,
            fit_intercept=False,
        )
        self.clf.fit(X_train_scaled, y_train, sample_weight=sw_train)
        self.directions = torch.tensor(self.clf.coef_.reshape(n_layers, n_features), dtype=torch.float32)

        if X_val_scaled is not None and y_val is not None and len(np.unique(y_val)) > 1:
            from sklearn.metrics import roc_auc_score
            val_scores = self.clf.decision_function(X_val_scaled)
            self.val_auroc = float(roc_auc_score(y_val, val_scores))
            self.val_margin = float(val_scores[y_val == 1].mean() - val_scores[y_val == 0].mean())
            self.val_n_total = int(len(y_val))
            self.val_n_honest = int((y_val == 0).sum())
            self.val_n_deceptive = int((y_val == 1).sum())

    def score(self, activations: Activations) -> Scores:
        if self.directions is None:
            raise RuntimeError("Detector not fitted")

        masked = activations.get_masked_activations()
        if masked.numel() == 0:
            return Scores([], [])

        if activations.acts_are_masked_only:
            n_tok = int(masked.shape[0])
            if activations.detect_token_count_per_dialogue is None:
                raise ValueError(
                    "Masked-only activations require detect_token_count_per_dialogue."
                )

            expected = int(sum(activations.detect_token_count_per_dialogue))
            if expected != n_tok:
                raise ValueError(
                    f"Masked-only token count mismatch: expected {expected}, got {n_tok}."
                )

            if activations.dialogue_index_per_token is not None:
                if int(activations.dialogue_index_per_token.shape[0]) != n_tok:
                    raise ValueError(
                        "Masked-only dialogue_index_per_token length mismatch: "
                        f"{int(activations.dialogue_index_per_token.shape[0])} != {n_tok}."
                    )

                n_d = len(activations.detect_token_count_per_dialogue)
                idx = activations.dialogue_index_per_token.to(torch.long).cpu()

                if idx.numel() and (idx.min().item() < 0 or idx.max().item() >= n_d):
                    raise ValueError(
                        f"dialogue_index_per_token has out-of-range indices "
                        f"(min={idx.min().item()}, max={idx.max().item()}, n_dialogues={n_d})."
                    )

                got = torch.bincount(idx, minlength=n_d).tolist()
                exp = [int(x) for x in activations.detect_token_count_per_dialogue]
                if got != exp:
                    raise ValueError(
                        "Masked-only metadata mismatch: "
                        f"bincount={got}, counts={exp}"
                    )

        if self.scaler_mean is not None and self.scaler_scale is not None:
            masked = (
                masked.float() - self.scaler_mean.to(masked.device)
            ) / self.scaler_scale.to(masked.device)

        scores_tensor = (masked * self.directions.to(masked.device)).sum(dim=-1).mean(dim=-1)

        if activations.detect_token_count_per_dialogue is not None:
            n_per_dialogue = activations.detect_token_count_per_dialogue
        else:
            n_per_dialogue = activations.tokenized_dataset.detection_mask.sum(dim=1).tolist()

        scores_list = list(torch.split(scores_tensor.cpu(), n_per_dialogue))

        if activations.labels is not None:
            if len(activations.labels) != len(scores_list):
                raise ValueError(
                    f"activations.labels length {len(activations.labels)} != "
                    f"n_dialogues {len(scores_list)}"
                )
            labels_out = list(activations.labels)
        else:
            warnings.warn(
                "No dialogue labels provided to detector.score(); returning Label.SKIP.",
                RuntimeWarning,
            )
            labels_out = [Label.SKIP] * len(scores_list)

        return Scores(scores_list, labels_out)

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "layers": self.layers,
                    "reg_coeff": self.reg_coeff,
                    "scale": self.scale,
                    "train_data": self.train_data,
                    "clf": self.clf,
                    "scaler_mean": self.scaler_mean.cpu() if self.scaler_mean is not None else None,
                    "scaler_scale": self.scaler_scale.cpu() if self.scaler_scale is not None else None,
                    "directions": self.directions.cpu() if self.directions is not None else None,
                    "val_auroc": self.val_auroc,
                    "val_margin": self.val_margin,
                    "val_n_total": self.val_n_total,
                    "val_n_honest": self.val_n_honest,
                    "val_n_deceptive": self.val_n_deceptive,
                },
                f,
            )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LogisticRegressionDetector":
        with open(path, "rb") as f:
            data = pickle.load(f)

        if "reg_coeff" not in data or data["reg_coeff"] is None:
            raise RuntimeError(
                f"Probe checkpoint missing 'reg_coeff' — refusing to use a silent default. "
                "Retrain the probe to stamp it."
            )
        det = cls(
            layers=data["layers"],
            reg_coeff=float(data["reg_coeff"]),
            scale=data.get("scale", True),
            train_data=data.get("train_data"),
        )
        det.clf = data.get("clf")
        det.scaler_mean = data.get("scaler_mean")
        det.scaler_scale = data.get("scaler_scale")
        det.directions = data.get("directions")
        det.val_auroc = data.get("val_auroc")
        det.val_margin = data.get("val_margin")
        det.val_n_total = data.get("val_n_total")
        det.val_n_honest = data.get("val_n_honest")
        det.val_n_deceptive = data.get("val_n_deceptive")
        return det

    def effective_direction(self, layer: int) -> Tensor:
        """Probe direction in activation-space coordinates: ``w_eff = w_scaled / sigma`` (raw ``directions`` if scale=False)."""
        if self.directions is None:
            raise RuntimeError("Detector not fitted; no directions to return.")
        if layer not in self.layers:
            raise KeyError(f"Layer {layer} not in this detector's layers {self.layers!r}.")
        idx = self.layers.index(layer)
        w = self.directions[idx].float()
        if not self.scale:
            return w
        if self.scaler_scale is None:
            raise RuntimeError(
                "Probe was fit with scale=True but scaler_scale is None — refusing to return "
                "effective_direction for an un-unscalable legacy probe."
            )
        sigma = self.scaler_scale[idx].float()
        return w / (sigma + 1e-12)

    def assert_score_invariance(self, raw_acts: Tensor, layer: int, *, atol: float = 1e-3) -> None:
        """Assert ``score_scaled(x)`` and ``<x, w_eff>`` agree up to a probe-only constant offset."""
        if self.directions is None:
            raise RuntimeError("Detector not fitted.")
        if layer not in self.layers:
            raise KeyError(f"Layer {layer} not in {self.layers!r}.")
        if raw_acts.dim() != 2:
            raise ValueError(f"raw_acts must be 2D [N, hidden]; got shape {tuple(raw_acts.shape)}.")
        idx = self.layers.index(layer)
        w_eff = self.effective_direction(layer).to(raw_acts.device).float()
        if self.scale:
            mu = self.scaler_mean[idx].to(raw_acts.device).float()
            sigma = self.scaler_scale[idx].to(raw_acts.device).float()
            w_scaled = self.directions[idx].to(raw_acts.device).float()
            score_a = (((raw_acts.float() - mu) / (sigma + 1e-12)) * w_scaled).sum(dim=-1)
        else:
            score_a = (raw_acts.float() * self.directions[idx].to(raw_acts.device).float()).sum(dim=-1)
        score_b = (raw_acts.float() * w_eff).sum(dim=-1)
        diff = score_a - score_b
        diff_var = float(diff.var().item()) if diff.numel() > 1 else 0.0
        if diff_var > atol:
            raise AssertionError(
                f"Score invariance broken at layer={layer}: var(scaled - unscaled) = {diff_var:.3e} > atol={atol:.3e}."
            )
        if self.scale:
            expected_offset = -(self.scaler_mean[idx].to(raw_acts.device).float() * w_eff).sum().item()
            observed_offset = float(diff.mean().item())
            offset_err = abs(expected_offset - observed_offset)
            if offset_err > atol:
                raise AssertionError(
                    f"Offset mismatch at layer={layer}: observed {observed_offset:.6f} vs expected {expected_offset:.6f} (|diff|={offset_err:.3e})."
                )

    def assert_trained_on(self, expected_train_data: str) -> None:
        """Raise if this probe was not trained on ``expected_train_data``.
        """
        if self.train_data is None:
            warnings.warn(
                "Probe has no train_data metadata (legacy checkpoint). "
                f"Skipping integrity check; expected train_data={expected_train_data!r}.",
                RuntimeWarning,
            )
            return
        if str(self.train_data).strip() != str(expected_train_data).strip():
            raise RuntimeError(
                "Probe train_data mismatch: probe was trained on "
                f"{self.train_data!r} but the current run expected {expected_train_data!r}. "
                "This usually means a stale per-language probe checkpoint shadowed a fresh sweep — "
                "delete the checkpoint and re-run the sweep, or pass --train-data correctly."
            )


class MMSDetector(Detector):
    """Mean-difference detector on one layer."""

    def __init__(self, layers: List[int], layer_index: Optional[int] = None):
        self.layers = layers
        self.layer_index = layer_index or (layers[len(layers) // 2] if layers else 0)
        self.direction: Optional[Tensor] = None

    def _get_layer_acts(self, activations: Activations) -> Tensor:
        masked = activations.get_masked_activations()
        idx = activations.layers.index(self.layer_index) if self.layer_index in activations.layers else 0
        return masked[:, idx, :].float()

    def fit(
        self,
        honest_activations: Activations,
        deceptive_activations: Activations,
    ) -> None:
        mean_h = self._get_layer_acts(honest_activations).mean(dim=0).cpu()
        mean_d = self._get_layer_acts(deceptive_activations).mean(dim=0).cpu()
        self.direction = mean_d - mean_h

    def score(self, activations: Activations) -> Scores:
        if self.direction is None:
            raise RuntimeError("Detector not fitted")

        acts = self._get_layer_acts(activations).cpu()
        sc = (acts * self.direction.unsqueeze(0)).sum(dim=1)

        if activations.acts_are_masked_only:
            n_tok = int(sc.shape[0])
            if activations.detect_token_count_per_dialogue is None:
                raise ValueError(
                    "Masked-only activations require detect_token_count_per_dialogue."
                )

            expected = int(sum(activations.detect_token_count_per_dialogue))
            if expected != n_tok:
                raise ValueError(
                    f"Masked-only token count mismatch: expected {expected}, got {n_tok}."
                )

            if activations.dialogue_index_per_token is not None:
                if int(activations.dialogue_index_per_token.shape[0]) != n_tok:
                    raise ValueError(
                        "Masked-only dialogue_index_per_token length mismatch: "
                        f"{int(activations.dialogue_index_per_token.shape[0])} != {n_tok}."
                    )

                n_d = len(activations.detect_token_count_per_dialogue)
                idx = activations.dialogue_index_per_token.to(torch.long).cpu()

                if idx.numel() and (idx.min().item() < 0 or idx.max().item() >= n_d):
                    raise ValueError(
                        f"dialogue_index_per_token has out-of-range indices "
                        f"(min={idx.min().item()}, max={idx.max().item()}, n_dialogues={n_d})."
                    )

                got = torch.bincount(idx, minlength=n_d).tolist()
                exp = [int(x) for x in activations.detect_token_count_per_dialogue]
                if got != exp:
                    raise ValueError(
                        "Masked-only metadata mismatch: "
                        f"bincount={got}, counts={exp}"
                    )

        if activations.detect_token_count_per_dialogue is not None:
            n_per_dialogue = activations.detect_token_count_per_dialogue
        else:
            n_per_dialogue = activations.tokenized_dataset.detection_mask.sum(dim=1).tolist()

        scores_list = list(torch.split(sc, n_per_dialogue))

        if activations.labels is not None:
            if len(activations.labels) != len(scores_list):
                raise ValueError(
                    f"activations.labels length {len(activations.labels)} != "
                    f"n_dialogues {len(scores_list)}"
                )
            labels_out = list(activations.labels)
        else:
            warnings.warn(
                "No dialogue labels provided to detector.score(); returning Label.SKIP.",
                RuntimeWarning,
            )
            labels_out = [Label.SKIP] * len(scores_list)

        return Scores(scores_list, labels_out)

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "layers": self.layers,
                    "layer_index": self.layer_index,
                    "direction": self.direction,
                },
                f,
            )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MMSDetector":
        with open(path, "rb") as f:
            data = pickle.load(f)

        det = cls(layers=data["layers"], layer_index=data.get("layer_index"))
        det.direction = data["direction"]
        return det