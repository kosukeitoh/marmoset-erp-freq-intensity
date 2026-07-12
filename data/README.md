# Marmoset Auditory ERP — Frequency × Intensity Dataset

BIDS-formatted EEG dataset from common marmosets (*Callithrix jacchus*) recorded during passive listening to pure tones varying in frequency (125–16000 Hz, 8 levels) and intensity (45, 60, 75 dB SPL).

Analysis code and full documentation are available on GitHub:
**https://github.com/kosukeitoh/marmoset-erp-freq-intensity**

## Structure

- `marmoset_FreqIntensity/` — BIDS root
  - `dataset_description.json`
  - `participants.tsv`
  - `sub-Cj399/`
  - `sub-Cj459/`
  - `derivatives/`
    - `erp-pipeline-ref-linkedEars/`
    - `erp-pipeline-ref-CAR/`
    - `erp-pipeline-ref-CMR/`

## License

CC BY 4.0 — see GitHub repository for details.
