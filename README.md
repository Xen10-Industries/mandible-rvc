# Mandible RVC Trainer Sidecar

Headless RVC v2 (Retrieval-based Voice Conversion) deep-learning vocal model trainer and ONNX exporter for [Mandible](https://github.com/Xen10-Industries/Mandible).

This tool trains 40M-parameter neural acoustic voice models (VITS + ContentVec + RMVPE) from vocal audio recordings and exports them directly into `.onnx` and `.index` packages ready for real-time inference inside Mandible.

---

## Features

- **Headless CLI:** Run training jobs directly from terminal or via Mandible's GUI with zero WebUI bloat.
- **Automated Slicing & Preprocessing:** Automatic silence removal, loudness normalization, and STFT feature prep.
- **ContentVec & RMVPE:** High-precision speech-to-phoneme representations with robust pitch contour tracking.
- **Direct ONNX Export:** Automatically exports the final VITS generator into an optimized `.onnx` model and `.index` feature retrieval codebook.
- **JSON Telemetry Stream:** Emits real-time training metrics over stdout (`epoch`, `loss`, `eta`) so host GUIs can update natively.

---

## Installation & Setup

### Requirements
- Windows 10/11 x64 or Linux x64
- Python 3.10 – 3.12 (Python 3.10/3.11 recommended for CUDA PyTorch)
- NVIDIA GPU with CUDA support (e.g. RTX 3060/4070 or Google Colab T4/A100)

```powershell
git clone https://github.com/Xen10-Industries/mandible-rvc.git
cd mandible-rvc
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## CLI Usage

### Train a Custom Voice Model
```powershell
python -m mandible_rvc train `
    --input "C:\path\to\vocals" `
    --name "Xen Overmind" `
    --epochs 200 `
    --batch-size 8 `
    --f0-method rmvpe `
    --output-dir "%APPDATA%\Xen10\Mandible\ai_models"
```

The resulting `xen_overmind.onnx`, `xen_overmind.index`, and `xen_overmind.json` are written directly into Mandible's voice library, ready to select in the **AI Voice Changer** rack plugin.

---

## License

This project is open-source under the [MIT License](LICENSE). Based on work by the [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).
