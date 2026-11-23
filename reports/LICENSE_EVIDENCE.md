# Dataset and model license evidence

Checked 2026-11-24. This is provenance documentation, not legal advice.

## Dataset

- Current Hugging Face page: https://huggingface.co/datasets/AI-Lab-Makerere/beans
- Pinned card: https://huggingface.co/datasets/AI-Lab-Makerere/beans/blob/27aa014ce09b193e1a6f58112d4a66e0eddb69c5/README.md
- Dataset revision: `27aa014ce09b193e1a6f58112d4a66e0eddb69c5`.
- The card YAML identifies the license as MIT. Its prose Licensing Information subsection remains incomplete, so that subsection alone is insufficient evidence.
- Current upstream data README: https://github.com/AI-Lab-Makerere/ibean/blob/master/README.md . Its Versions/Release table explicitly assigns MIT to the data and credits Makerere AI Lab. It also describes collaboration with NaCRRI and expert field annotation.
- Upstream license: https://github.com/AI-Lab-Makerere/ibean/blob/master/LICENSE . MIT, copyright 2020 AIR Lab Makerere University. A verbatim copy is included in `DATASET_LICENSE.txt` to preserve the required notice for image excerpts.
- GitHub license endpoint independently classifies the file MIT: https://api.github.com/repos/AI-Lab-Makerere/ibean/license . Observed LICENSE blob SHA: `4d27d05aff21ea38d5dc4a621fa6ecb21f763067`.
- Citation supplied by the HF card: Makerere AI Lab, Bean disease dataset, January 2020, https://github.com/AI-Lab-Makerere/ibean/ .
- The upstream aggregate count (1,296) differs by one from the actual pinned HF split total (1,295). This project uses the pinned HF parquet files and records their SHA-256 values and exact download URLs in `data_manifest.json`.

## Model

- Current card: https://huggingface.co/microsoft/resnet-18
- Pinned card: https://huggingface.co/microsoft/resnet-18/blob/65a5785d9156231087c481e0c7dd33a5ff6f7e3e/README.md
- Model revision: `65a5785d9156231087c481e0c7dd33a5ff6f7e3e`.
- Card metadata identifies Apache-2.0 and ImageNet-1k pretraining. The card attributes the architecture to the ResNet paper and notes that Hugging Face wrote the card.
- Paper: https://arxiv.org/abs/1512.03385
- Original architecture repository: https://github.com/KaimingHe/deep-residual-networks
- Apache-2.0 text: https://www.apache.org/licenses/LICENSE-2.0

The code's MIT license is separate. Model weights and the complete dataset are downloaded on demand and are not shipped in this source archive. Only the attributed dataset image excerpts needed for the failure and explanation panels are included.
