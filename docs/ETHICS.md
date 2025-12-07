# Ethics and Data Governance Statement

## Institutional Review Board (IRB) Status

This research uses **exclusively publicly available, de-identified datasets**
and does not involve human subjects research requiring IRB approval.

### CholecSeg8k Dataset
- **Source**: Kaggle (public platform)
- **Original Study**: Cholec80 dataset from IRCAD (Strasbourg, France)
- **De-identification**: All videos are fully anonymized with no patient identifiers
- **License**: Available for research purposes per Kaggle terms

## HIPAA Compliance

This project does **not** process, store, or transmit Protected Health Information (PHI).
All data used is:
- Pre-anonymized at source
- Contains no direct or indirect patient identifiers
- Obtained from public repositories

## Intended Use

This software is intended for **research and educational purposes only**.

### NOT Intended For:
- Clinical diagnosis or treatment decisions
- Real-time intraoperative guidance without additional validation
- Deployment as a medical device without regulatory approval

### Regulatory Pathway (Future Work)

Any clinical application would require:
1. FDA 510(k) clearance (USA) or CE marking (EU)
2. Prospective clinical validation study
3. IRB approval for human subjects research
4. HIPAA-compliant data handling infrastructure

## Data Handling Practices

1. **No data is stored in this repository** - datasets must be downloaded separately
2. **Outputs contain no patient data** - only model weights and synthetic visualizations
3. **Pipeline designed for air-gapped environments** - can run on isolated systems

## Contact

For questions about data governance or ethical use:
- Email: maximilian.dressler@stud.uni-heidelberg.de
- Institution: Heidelberg University
