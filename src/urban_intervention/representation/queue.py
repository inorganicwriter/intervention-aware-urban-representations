"""Response-aware FIFO queue for supervised contrastive learning."""

from __future__ import annotations

from typing import TypedDict

import torch


class LabeledQueueState(TypedDict):
    embeddings: torch.Tensor
    responses: torch.Tensor
    response_mask: torch.Tensor
    response_se: torch.Tensor
    ids: torch.Tensor


class EmbeddingQueue:
    """Sample-level FIFO queue of detached embeddings and response metadata.

    The trainer supplies response labels and unit identifiers so queued samples
    can be classified as positives, negatives, incomparable pairs, or duplicate
    views of the same unit.  Tensors are kept on CPU and only the newest
    ``capacity`` rows are retained.
    """

    def __init__(
        self,
        dim: int,
        capacity: int,
        device: torch.device | None = None,
    ):
        if capacity < 0:
            raise ValueError(f"Queue capacity must be >= 0, got {capacity}")
        self.dim = dim
        self.capacity = capacity
        # Retained for API compatibility. Queue storage deliberately remains
        # on CPU to avoid persistent accelerator-memory pressure.
        self.device = device
        self._embeddings: torch.Tensor | None = None
        self._responses: torch.Tensor | None = None
        self._response_mask: torch.Tensor | None = None
        self._response_se: torch.Tensor | None = None
        self._ids: torch.Tensor | None = None

    def enqueue(
        self,
        embeddings: torch.Tensor,
        responses: torch.Tensor | None = None,
        response_mask: torch.Tensor | None = None,
        response_se: torch.Tensor | None = None,
        ids: torch.Tensor | None = None,
    ) -> None:
        """Append rows and evict exactly the oldest excess samples."""
        if self.capacity <= 0 or embeddings.shape[0] == 0:
            return
        detached = embeddings.detach().cpu()
        if detached.ndim != 2 or detached.shape[1] != self.dim:
            actual = detached.shape[1] if detached.ndim == 2 else tuple(detached.shape)
            raise ValueError(f"Queue dim {self.dim} does not match embeddings dim {actual}")

        metadata = (responses, response_mask, response_se, ids)
        supplied = [value is not None for value in metadata]
        if any(supplied) and not all(supplied):
            raise ValueError("Queue response metadata and ids must be supplied together")
        labeled = all(supplied)
        currently_labeled = self._responses is not None
        if self._embeddings is not None and labeled != currently_labeled:
            raise ValueError("Cannot mix labeled and unlabeled entries in one queue")

        self._embeddings = self._append(self._embeddings, detached)
        if labeled:
            assert responses is not None
            assert response_mask is not None
            assert response_se is not None
            assert ids is not None
            n = detached.shape[0]
            if any(value.shape[0] != n for value in (responses, response_mask, response_se, ids)):
                raise ValueError("Queue metadata rows must align with embeddings")
            self._responses = self._append(self._responses, responses.detach().cpu())
            self._response_mask = self._append(
                self._response_mask, response_mask.detach().cpu()
            )
            self._response_se = self._append(self._response_se, response_se.detach().cpu())
            self._ids = self._append(self._ids, ids.detach().cpu())

    def _append(self, current: torch.Tensor | None, incoming: torch.Tensor) -> torch.Tensor:
        combined = incoming if current is None else torch.cat([current, incoming], dim=0)
        return combined[-self.capacity :]

    def state(self) -> torch.Tensor | None:
        """Current embedding rows in insertion order, or ``None`` when empty."""
        return self._embeddings

    def labeled_state(self) -> LabeledQueueState | None:
        """Return aligned response metadata when this is a labeled queue."""
        if self._embeddings is None or self._responses is None:
            return None
        assert self._response_mask is not None
        assert self._response_se is not None
        assert self._ids is not None
        return {
            "embeddings": self._embeddings,
            "responses": self._responses,
            "response_mask": self._response_mask,
            "response_se": self._response_se,
            "ids": self._ids,
        }

    def __len__(self) -> int:
        return 0 if self._embeddings is None else self._embeddings.shape[0]
