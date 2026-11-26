# Zipper / Karvish — Model & Asset Manifest

## 1. Bundled Runtime Assets (Included via Git LFS)

| Asset Path | Size | Role | Runtime Engine |
| --- | ---: | --- | --- |
| `app/src/main/assets/models/karvish_v62.onnx` | 54.42 MB | Primary V62 Unified `(K+1)` FP16 UI Action Policy | `OnDeviceRankerEngine.kt` |
| `app/src/main/assets/models/icon_visual_embeddings.bin` | 0.03 MB | 512-D visual icon embeddings | `OnDeviceRankerEngine.kt` |
| `app/src/main/assets/models/vocab.txt` | 0.25 MB | Extended 32,522-token WordPiece vocabulary | `KarvishWordPieceTokenizer.kt` |
| `app/src/main/assets/models/tokenizer_vocab.json` | 0.49 MB | WordPiece JSON vocabulary | `OnDeviceRankerEngine.kt` |
| `app/src/main/assets/models/tokenizer.model` | 0.22 MB | SentencePiece tokenizer | `KarvishSentencePieceTokenizer.kt` |
| `app/src/main/assets/models/zipper_core.glb` | 4.28 MB | 3D Core Orb mesh | `Zipper3DCoreView.kt` |
| `app/src/main/jniLibs/arm64-v8a/libneedle.so` | 14.14 MB | Cactus Compute Needle 2 ARM64 binary | `Needle2NativeEngine.kt` |
| `app/src/main/assets/needle/needle_lora_v3.safetensors` | 7.63 MB | Needle 2 V3 LoRA adapter | `Needle2NativeEngine.kt` |
| `app/src/main/assets/voice/paradee_8m.onnx` | 8.62 MB | Paradee-8M Kokoro StyleTTS2 INT8 TTS | `ParadeeVoiceEngine.kt` |
| `app/src/main/assets/voice/speaker_embedding.onnx` | 6.59 MB | 3D-Speaker / Cam++ 192-D voiceprint verifier | `SpeakerVoiceVerifier.kt` |

## 2. Optional / External Assets (Not in GitHub)

- **Retraining Checkpoints & Datasets (~2.08 GB):** `v62_best.pt`, `v62_authoritative_dataset.pt`, `v54_universal_search_best.pt`, `v52_locked_gold.pt` — share separately via private cloud storage if continuing PyTorch training.
- **Karvish-Web ONNX (`karvish_web_fp16.onnx`, 47.42 MB):** Re-exportable via `python tools/training/karvish_web/export_web_onnx.py` and optionally placed at `app/src/main/assets/models/karvish_web.onnx`.
- **Base HF Backbone:** `sentence-transformers/all-MiniLM-L6-v2` (auto-downloaded by `transformers` during training/export).
- **Public Datasets:** AndroidControl / AMEX (re-downloadable via `tools/training/prepare_android_control.py`).
