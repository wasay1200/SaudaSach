# Data provenance and permissions

## Urdu speech

- Provider: CSALT, Lahore University of Management Sciences.
- Dataset: https://huggingface.co/datasets/CSALT/deepfake_detection_dataset_urdu
- Repository: https://github.com/CSALT-LUMS/urdu-deepfake-dataset
- Frozen dataset revision: `eb8f16623108324867e4424a646937409c64b82b`.
- The downloaded dataset README explicitly states **CC BY-NC 4.0** for data hosted on Hugging Face and Google Drive. Preserve provider attribution and noncommercial restrictions. `docs/Urdu_dataset_README.md` records the upstream notice; manifests identify file hashes and provenance.
- Genuine and provided synthetic recordings are used as released. The pipeline does not create new cloned identities. A public dataset license is recorded as the source permission; the project does not claim access to original individual consent forms or independent validation of provider collection procedures.

## Environmental noise

- Provider: Karol J. Piczak, ESC-50 dataset.
- Source: https://github.com/karoldvl/ESC-50
- Revision recorded in `data/noise/source.json`.
- Dataset README states **CC BY-NC 3.0**, with a CC BY subset indicated in metadata. We conservatively apply the noncommercial corpus restriction to the experiment.
- Forty selected files span rain, vacuum cleaner, engine and sea waves. Original file/source IDs and hashes are in `data/noise/manifest.csv`. Upstream notices and attributions are retained in `docs/ESC50_source_README.md` and `docs/ESC50_source_LICENSE.txt`.
- ESC folds 1–3 supply training noise, fold 4 development, fold 5 test. Original-source IDs are checked for overlap.

## Distribution

Raw audio and derived feature caches are excluded from Git. Download scripts retrieve the originals from their providers. Public dataset-relative paths identify research records, not app-user files or participant contacts. Trained artifacts are provided for noncommercial research review subject to applicable upstream rights. Source code and dataset/weight rights are distinct; no claim is made that the trained weights remove the source restrictions.
