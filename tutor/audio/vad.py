"""Silero VAD v5 (ONNX, ~2 MB, <1 ms por trama de 32 ms en CPU)."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # 32 ms a 16 kHz (tamaño que exige Silero v5)
_CONTEXT = 64  # Silero v5 concatena las últimas 64 muestras de la trama anterior


class SileroVAD:
    """VAD con estado. Una instancia por sesión (el estado recurrente es por flujo de audio)."""

    def __init__(self, session: ort.InferenceSession):
        self._session = session
        self.reset()

    @staticmethod
    def load_session(path: str) -> ort.InferenceSession:
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Probabilidad de voz (0..1) para una trama float32 de 512 muestras."""
        if frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(f"La trama debe tener {FRAME_SAMPLES} muestras")
        x = np.concatenate([self._context, frame]).astype(np.float32)[None, :]
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)}
        )
        self._context = frame[-_CONTEXT:].astype(np.float32)
        return float(out[0][0])
