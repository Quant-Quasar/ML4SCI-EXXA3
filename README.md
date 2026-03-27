# ML4SCI EXXA3 - Evaluation Test
### GSoC 2026 | Exoplanet Atmosphere Characterisation

**Project**: EXXA3: ML for Exoplanet Atmosphere Characterisation

---

## Overview

This repository contains the complete evaluation test submission for the ML4SCI EXXA3 GSoC 2026 project. Two tasks are implemented in a single end-to-end notebook:

| Task | Method | Key Metric |
|------|--------|------------|
| General: Unsupervised clustering of ALMA protoplanetary disk images | Convolutional Autoencoder + UMAP + K-means | Silhouette = 0.628, MSE = 0.007042 |
| Sequential: Transit light curve binary classification | 1D Residual CNN | Test AUC = 0.7995 |

> **Note**: Silhouette scores and Spearman correlations will vary slightly across runs due to stochastic elements in KMeans initialisation and UMAP. Reported values are from the final complete run.

---

## Repository Structure

```
ML4SCI-EXXA3/
├── EXXA_3_Test_Task.ipynb         # Complete notebook (run start to finish)
├── README.md
└── results/
    ├── master_summary.png          # Full pipeline summary figure
    ├── umap_clusters.png           # UMAP cluster visualisation
    ├── cluster_representatives.png # 5 images closest to each cluster centroid
    ├── radial_profiles.png         # Azimuthal brightness profiles per cluster
    ├── ae_reconstructions.png      # Autoencoder input vs reconstruction
    ├── disk_samples.png            # Sample disk images per planet-count range
    ├── clf_training_curves.png     # Classifier training loss and validation AUC
    ├── roc_evaluation.png          # ROC curve with per-tier breakdown
    ├── sample_lightcurves.png      # Sample transit and non-transit light curves
    ├── lc_param_distributions.png  # Physical parameter distributions
    ├── eda_planet_count_histogram.png
    ├── eda_mean_images_per_range.png
    ├── eda_intensity_vs_planets.png
    ├── eda_radius_vs_planets.png
    ├── eda_aspect_ratio_per_range.png
    └── eda_augmented_clustering_comparison.png
```

Pre-trained model weights (`best_ae.pt`, `best_clf.pt`) are hosted on Google Drive:
**[Download weights](https://drive.google.com/drive/folders/1_w_7FbExkJS65zE_LCcrPm8tE-gpzaqE?usp=sharing)**

---

## Running the Notebook

The notebook is designed for Google Colab (T4 GPU). Runtime is approximately 25 minutes.

1. Open `EXXA_3_Test_Task.ipynb` in Google Colab
2. Mount your Google Drive when prompted
3. Place `continuum_data_subset.zip` in your Drive root (dataset provided by ML4SCI mentors). Direct download via gdown is unreliable due to rate limits - Drive mount is the recommended approach.
4. Run `Runtime → Run All` - no further intervention required after Drive mount


To use pre-trained weights instead of retraining, download `best_ae.pt` and `best_clf.pt` from the Drive link above and place them in the Colab working directory.

---

## Inference on Withheld Data

Both models expose documented inference pipelines:

```python
# ALMA disk clustering
results = run_disk_inference(data_dir='path/to/fits/files')
# Returns: cluster labels, latent vectors, UMAP coordinates

# Transit light curve classification
probs, labels = run_transit_inference(X_new, clf_weights='best_clf.pt', threshold=0.94)
# X_new: np.ndarray (N, 500) -- raw flux arrays
# Returns: P(planet) per curve, binary predictions
```

Both functions include input validation and error handling. The threshold of 0.94 was determined from the validation set by maximising F1 score.

---

## Results Summary

### General Test: ALMA Disk Clustering

- **Dataset**: 150 synthetic ALMA continuum observations at 1250 μm (fits format, Stokes I)
- **Architecture**: Convolutional Autoencoder (1,368,305 params), two-phase training (MSE then MSE+SSIM)
- **Augmentation**: Random 0-360 degree rotation + horizontal/vertical flips (orientation invariance)
- **Clustering**: UMAP-guided K-means (k=6) - KMeans on 2D UMAP embedding rather than raw 64-dim latent space
- **Reconstruction MSE**: 0.007042
- **Silhouette (raw latent, k=6)**: 0.448
- **Silhouette (UMAP-guided, k=6)**: 0.628
- Radial brightness profiles per cluster connect unsupervised assignments to planet-carved gap signatures

**EDA findings**: Simple photometric features (mean pixel intensity, disk radius) show no significant correlation with planet count (Spearman rho < 0.1), confirming deep representation learning is necessary. A physics-feature augmentation experiment (disk radius + aspect ratio concatenated to latent vectors) reduced silhouette to 0.564, confirming the autoencoder latent space encodes more discriminative structure than hand-crafted features.

### Sequential Test: Transit Light Curve Classification

- **Dataset**: 10,000 synthetic light curves generated with `batman` (Kreidberg 2015)
- **Architecture**: 1D Residual CNN (340,513 params), label smoothing, gradient clipping, CosineAnnealingLR
- **Noise tiers**: Easy (50-200 ppm), Medium (200-500 ppm), Hard (500-1500 ppm)
- **Test AUC**: 0.7995
- **Test Accuracy**: 75.8%
- **Per-tier recall**: Easy 0.665 | Medium 0.604 | Hard 0.615
- **Matched filter baseline AUC**: 0.7320 (1D ResNet outperforms classical signal processing by 0.0675 AUC points)

Conservative classifier behaviour (high precision, lower recall) is scientifically appropriate for transit surveys, where false positives waste telescope follow-up time.

---

## Dependencies

```
torch
torchvision
numpy
scikit-learn
matplotlib
astropy
batman-package
umap-learn
pytorch-msssim
lightkurve
tqdm
```

All packages are available on Google Colab without additional configuration. Install with:

```bash
pip install batman-package astropy umap-learn tqdm pytorch-msssim lightkurve gdown
```

---

## References

- Terry et al. (2022). *Locating Hidden Exoplanets in ALMA Data Using Machine Learning.* ApJ 941(2):192. doi:10.3847/1538-4357/aca477
- Kreidberg (2015). *batman: BAsic Transit Model cAlculatioN in Python.* PASP 127(957):1161. doi:10.1086/683602
- Kanagawa et al. (2016). *Mass Constraint for a Planet in a Protoplanetary Disk from the Gap Width.* PASJ 68:43. doi:10.1093/pasj/psw037
- McInnes et al. (2018). *UMAP: Uniform Manifold Approximation and Projection.* arXiv:1802.03426

---
## AI Assistance Disclosure

AI tools were used for code review, documentation formatting, and literature 
citation verification. All experiments, model architectures, training runs, 
and results are the author's own original work.

---

*GSoC 2026 ML4SCI EXXA3 | adityaparashar3434@gmail.com* 