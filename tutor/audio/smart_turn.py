"""Smart Turn v3.2 (Pipecat, BSD-2): VAD semántico que estima si el hablante ha terminado su turno
analizando la forma de onda (entonación, ritmo, alargamientos), no el texto.

Modelo: encoder Whisper-tiny + cabeza lineal, 8M parámetros, ONNX int8 de 8 MB, ~15-25 ms en CPU.
Usamos el extractor log-mel de faster-whisper para no depender de `transformers`.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from faster_whisper.feature_extractor import FeatureExtractor

_WINDOW_S = 8
_SR = 16000
_N = _WINDOW_S * _SR


class SmartTurn:
    def __init__(self, path: str):
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 2
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self._fe = FeatureExtractor(feature_size=80, chunk_length=_WINDOW_S)

    def probability_complete(self, audio: np.ndarray) -> float:
        """Probabilidad (0..1) de que el turno esté completo, dado el audio del turno hasta ahora."""
        audio = np.asarray(audio, dtype=np.float32)[-_N:]
        if audio.shape[0] < _N:  # relleno con ceros AL PRINCIPIO, igual que en el entrenamiento
            audio = np.pad(audio, (_N - audio.shape[0], 0))
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        feats = self._fe(audio, padding=0)[:, :800].astype(np.float32)[None]
        return float(self._session.run(None, {"input_features": feats})[0].item())
