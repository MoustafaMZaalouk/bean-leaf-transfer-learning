# Local verification record

- Linux x86_64, Python 3.10.12, CPU only, four PyTorch threads.
- Built a new isolated virtual environment from `requirements.txt`. Installation succeeded, including the CPU PyTorch wheel.
- Downloaded data to a separate empty cache and recomputed all features with the pinned model. Ran complete train and evaluation paths. The full 128-image test prediction JSON was byte-identical to the earlier independently run experiment.
- Ran `make smoke` in the new environment. The real 18-image-per-split smoke pipeline, including partial fine-tuning and evaluation, passed. No synthetic smoke metrics are mixed into the report.
- Ran `make lint test`: lint and four-space formatting passed; all 13 correctness tests passed.
- Regenerated and visually inspected the confusion matrices, eight failure examples and three Grad-CAM panels. Labels were readable, panels complete and overlays aligned with processed images.
- GitHub Actions workflow is included but has not been executed on GitHub. No remote account or repository was changed.
- Runtime in `metrics.json` is a warm feature-cache measurement, not a clean-install or cold-download benchmark. It is intentionally not advertised as inference latency.
