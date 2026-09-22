"""The text tower: one query string in, one 768-d embedding out.

    query -> OpenCLIP basic_clean + canonicalize
          -> SigLIP2 tokenizer, padded/truncated to 64 tokens
          -> ONNX text tower
          -> L2-normalized 768-d embedding

The tokenizer is **not** an implementation detail. The image and text towers are
two halves of one model: a query tokenized with a different context length, or
without OpenCLIP's canonicalization, still yields a perfectly well-formed 768-d
vector that still ranks results — they are just subtly, unfalsifiably wrong,
which is the worst failure mode a search system has. That is why the
canonicalization below is reproduced byte-for-byte rather than approximated.

The vocabulary is multilingual (256k, Gemma). Vietnamese goes straight in;
no translation step is required or wanted.
"""

from __future__ import annotations

import html
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from siglip_infer.runtime import Device, ModelUnavailable, build_session, describe_io

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BUNDLE_ROOT / "models" / "siglip_text.onnx"
DEFAULT_TOKENIZER_PATH = BUNDLE_ROOT / "models" / "siglip_tokenizer.json"

#: Fixed at export. Must match the tower the image embeddings came from.
CONTEXT_LENGTH = 64
EMBEDDING_DIM = 768

#: SigLIP2 special-token ids. Checked on load, because a tokenizer with a
#: different EOS id pools the wrong token and produces a plausible embedding.
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1

UNIT_NORM_TOLERANCE = 1e-3

_ASCII_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def canonicalize(text: str) -> str:
    """Byte-for-byte equivalent to OpenCLIP's ``basic_clean`` + ``canonicalize``.

    Fixes mojibake, unescapes HTML twice (as OpenCLIP does), turns underscores
    into spaces, strips ASCII punctuation, lowercases, and collapses whitespace.
    """
    try:
        import ftfy
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ModelUnavailable("ftfy is not installed; see requirements.txt") from error
    cleaned = ftfy.fix_text(text)
    cleaned = html.unescape(html.unescape(cleaned)).strip()
    cleaned = cleaned.replace("_", " ").translate(_ASCII_PUNCTUATION_TABLE).lower()
    return " ".join(cleaned.split()).strip()


class SiglipTokenizer:
    """The exported SigLIP2 tokenizer, pinned to this tower's contract."""

    def __init__(
        self,
        path: Path | str = DEFAULT_TOKENIZER_PATH,
        *,
        context_length: int = CONTEXT_LENGTH,
    ) -> None:
        self._path = Path(path)
        self._context_length = context_length
        self._tokenizer: Any = None

    @property
    def context_length(self) -> int:
        return self._context_length

    def load(self) -> SiglipTokenizer:
        if self._tokenizer is not None:
            return self
        if not self._path.is_file():
            raise ModelUnavailable(f"tokenizer not found at {self._path}")
        try:
            from tokenizers import Tokenizer
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ModelUnavailable("tokenizers is not installed; see requirements.txt") from error

        tokenizer = Tokenizer.from_file(str(self._path))
        expected = {"<pad>": PAD_TOKEN_ID, "<eos>": EOS_TOKEN_ID}
        actual = {token: tokenizer.token_to_id(token) for token in expected}
        if actual != expected:
            raise ModelUnavailable(
                f"tokenizer special-token ids disagree with SigLIP: {actual}, expected {expected}"
            )
        tokenizer.enable_padding(
            length=self._context_length, pad_id=PAD_TOKEN_ID, pad_token="<pad>"
        )
        tokenizer.enable_truncation(max_length=self._context_length)
        self._tokenizer = tokenizer
        return self

    def encode(self, texts: str | Sequence[str]) -> NDArray[np.int64]:
        """Canonicalize and tokenize to a fixed ``(N, 64)`` int64 array."""
        if self._tokenizer is None:
            self.load()
        queries = [texts] if isinstance(texts, str) else list(texts)
        if not queries:
            raise ValueError("cannot tokenize an empty batch")

        rows: list[list[int]] = []
        for query in queries:
            if not isinstance(query, str):
                raise TypeError(f"query must be a string, got {type(query).__name__}")
            canonical = canonicalize(query)
            if not canonical:
                # Canonicalization strips punctuation, so "???" becomes "".
                # Embedding that would return the pad-only vector for every such
                # query, silently ranking an arbitrary set of people.
                raise ValueError(f"query {query!r} is empty after canonicalization")
            encoded = self._tokenizer.encode(canonical)
            ids = list(encoded.ids)
            if len(ids) != self._context_length:
                raise ModelUnavailable(
                    f"tokenizer produced {len(ids)} ids, expected {self._context_length}"
                )
            # ⚠️ EOS must terminate real content. Padding-only output satisfies
            # the shape contract and still moves text pooling to the wrong token.
            if EOS_TOKEN_ID not in ids:
                raise ModelUnavailable(f"tokenizer output for {query!r} has no SigLIP EOS token")
            rows.append(ids)
        return np.asarray(rows, dtype=np.int64)


class TextEncoder:
    """The SigLIP text tower on ONNX Runtime.

    Loaded once and reused; the session is safe to call from several threads.

        encoder = TextEncoder()
        vector = encoder.encode("người mặc áo đỏ")   # (768,) float32, unit norm
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        *,
        tokenizer_path: Path | str = DEFAULT_TOKENIZER_PATH,
        device: Device | str = Device.AUTO,
        tokenizer: SiglipTokenizer | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._tokenizer = (tokenizer or SiglipTokenizer(tokenizer_path)).load()
        self._session = build_session(
            self._model_path, device=device, intra_op_threads=intra_op_threads
        )
        self._validate_contract()

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def tokenizer(self) -> SiglipTokenizer:
        return self._tokenizer

    def _validate_contract(self) -> None:
        io = describe_io(self._session)
        inputs, outputs = io["inputs"], io["outputs"]
        if len(inputs) != 1 or inputs[0]["name"] != "text":
            raise ModelUnavailable(f"text model must take one input named 'text'; got {inputs}")
        if inputs[0]["type"] != "tensor(int64)":
            raise ModelUnavailable(f"text input must be int64; got {inputs[0]['type']}")
        shape = inputs[0]["shape"]
        if len(shape) != 2 or shape[1] != CONTEXT_LENGTH:
            raise ModelUnavailable(
                f"text input must be (batch, {CONTEXT_LENGTH}); got {shape}. "
                "A different context length silently changes every query embedding."
            )
        if len(outputs) != 1 or outputs[0]["name"] != "embedding":
            raise ModelUnavailable(
                f"text model must output one tensor named 'embedding'; got {outputs}"
            )
        output_shape = outputs[0]["shape"]
        if len(output_shape) != 2 or output_shape[1] != EMBEDDING_DIM:
            raise ModelUnavailable(
                f"text output must be (batch, {EMBEDDING_DIM}); got {output_shape}"
            )

    def encode(self, text: str) -> NDArray[np.float32]:
        """Encode one query to an L2-normalized 768-d vector."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str], *, batch_size: int = 16) -> NDArray[np.float32]:
        """Encode many queries to an ``(N, 768)`` array of unit vectors."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        tokens = self._tokenizer.encode(list(texts))
        chunks = [
            np.asarray(
                self._session.run(
                    ["embedding"],
                    {"text": np.ascontiguousarray(tokens[start : start + batch_size])},
                )[0],
                dtype=np.float32,
            )
            for start in range(0, len(tokens), batch_size)
        ]
        embeddings = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if embeddings.shape != (len(tokens), EMBEDDING_DIM):
            raise ModelUnavailable(
                f"text tower returned shape {embeddings.shape}, "
                f"expected {(len(tokens), EMBEDDING_DIM)}"
            )
        if not np.all(np.isfinite(embeddings)):
            raise ModelUnavailable("text tower produced non-finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(np.abs(norms - 1.0) > UNIT_NORM_TOLERANCE):
            # ⚠️ Normalization is inside the exported graph. Repairing it here
            # would hide a text/image model-contract mismatch.
            raise ModelUnavailable(
                f"text tower produced a non-unit embedding (norms {norms.tolist()})"
            )
        return np.ascontiguousarray(embeddings)

    def close(self) -> None:
        self._session = None

    def __enter__(self) -> TextEncoder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
