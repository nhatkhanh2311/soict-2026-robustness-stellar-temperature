# Robustness of Convolutional and Transformer-Based Stellar Temperature Estimators to Physical Spectral Artefacts

Code, trained checkpoints and result tables for the paper

> Nhat Khanh Pham and Minh Tuan Pham. *Robustness of Convolutional and Transformer-Based Stellar Temperature Estimators to Physical Spectral Artefacts.* SOICT 2026, Communications in Computer and Information Science, Springer (to appear).

Two parameter-matched effective-temperature estimators are trained on 188,972 SDSS DR19 BOSS spectra and compared under five physically motivated artefacts:

- **Conv-Pool**: a shared convolutional stem followed by three residual convolutional blocks and global average pooling (537,857 parameters)
- **Conv-Attn**: the same stem followed by a fixed sinusoidal positional encoding, three Pre-LN Transformer encoder layers and mean pooling (540,161 parameters)

The five artefacts are cosmic-ray spikes, occlusion, wavelength displacement, resolution degradation and interstellar reddening. Both models are equally accurate on clean data, so every result is reported as a degradation measured within the same model and seed, over five training seeds.

## Repository layout

```
.
├── models.py                  # Conv-Pool and Conv-Attn, the single definition used everywhere
├── common.py                  # checkpoint loading, prediction, metrics, test-split loading
├── 01_Data_Exploration_and_Filtering.ipynb
├── 02_Data_Alignment_and_Preprocessing.ipynb
├── 03_LDS_Weight_Generation.ipynb
├── 04a_Train_Baseline.ipynb   # one training run per seed
├── 04b_Baseline_Aggregation.ipynb
├── 05.0_Clean_Baseline.ipynb
├── 05.1_Cosmic_Ray_Spikes.ipynb
├── 05.2_Occlusion.ipynb
├── 05.3_Wavelength_Shift.ipynb
├── 05.4_Resolution_Degradation.ipynb
├── 05.5_Dust_Reddening.ipynb
├── 06_Paper_Figures.ipynb
├── data/
│   ├── sdss_all_star.zip      # the SDSS DR19 Astra catalogue query (compressed)
│   └── download_fits.py       # downloads the mwmStar spectra
├── checkpoints/               # {model}_seed{0..4}_best.pth, written by 04a
├── results/                   # tables and metadata written by 04a to 05.5
└── figures/                   # written by 06
```

Run every notebook from the repository root, so that `models.py`, `common.py`, `data/`, `results/` and `checkpoints/` resolve as relative paths.

## Environment

Python 3.10 or later with

```
torch  numpy  pandas  scipy  scikit-learn  h5py  astropy  matplotlib  seaborn  tqdm  jupyter
```

`papermill` is optional and only used to run the five training seeds non-interactively. The paper's models were trained on one NVIDIA T4 GPU. Evaluation also runs on a CPU, more slowly.

## Data

The spectra come from the public SDSS Data Release 19 (`mwm_astra` schema, v0.6.0). They are not redistributed here and are downloaded from the SDSS Science Archive Server.

**1. Unpack the catalogue.**

```bash
cd data
unzip sdss_all_star.zip        # -> data/sdss_all_star.csv
```

`sdss_all_star.csv` is the `lite_all_star` table of the `mwm_astra` schema, queried with `zwarning_flags = 0` (1,365,542 records).

**2. Download the spectra.**

```bash
python download_fits.py        # -> data/fits_files/mwmStar-0.6.0-<sdss_id>.fits
cd ..
```

The script fetches the `mwmStar` file of every catalogue entry into `data/fits_files/`, which is where notebook 02 looks for them. The full set is large, so check the free disk space before you start.

**3. Build the training tensor.** Run notebooks 01, 02 and 03 in order. They apply the five catalogue cuts (1,365,542 to 188,979 records), resample every spectrum on to a logarithmic grid of 4096 pixels over 3600 to 10,300 Å (77.0 km/s per pixel), drop 7 spectra with too few usable pixels (188,972 remain), split the catalogue 80/10/10 stratified in T<sub>eff</sub>, and compute the label-distribution-smoothing weights. They write

| File | Content |
|---|---|
| `data/sdss_catalog.csv` | the filtered catalogue (01) |
| `data/catalog_meta.json` | filter constants and the 47.9 K label-noise floor (01) |
| `data/sdss_tensors.h5` | all spectra, row *i* aligned with row *i* of the catalogue (02) |
| `data/sdss_final_catalog.csv` | the catalogue with its `dataset_split` column (02), and the `lds_weight` column (03) |
| `data/preprocessing_meta.json` | temperature scaling constants and the wavelength grid (02) |
| `data/lds_meta.json` | parameters of the label-distribution smoothing (03) |

## Reproducing the paper

| Step | Notebook | What it does | In the paper |
|---|---|---|---|
| 1 to 3 | `01`, `02`, `03` | catalogue cuts, resampling, partition, LDS weights | Section 3 |
| 4a | `04a_Train_Baseline` | trains both models for one seed; run for `SEED` = 0 to 4 | Section 4.2 |
| 4b | `04b_Baseline_Aggregation` | paired comparison across seeds, training cost, per-regime tests | Section 5 |
| 5.0 | `05.0_Clean_Baseline` | reloads every checkpoint, checks it reproduces its training RMSE to 0.01 K, writes the clean reference | Table 1 |
| 5.1 | `05.1_Cosmic_Ray_Spikes` | density and amplitude sweeps, $C = 1$ control, damage by spike position | Table 2, Fig. 2 |
| 5.2 | `05.2_Occlusion` | occlusion scan, profile-minimum control, width sweep by branch | Table 3 |
| 5.3 | `05.3_Wavelength_Shift` | sub-pixel and whole-token displacement | Table 4 |
| 5.4 | `05.4_Resolution_Degradation` | Gaussian blur over thirteen levels, continuum test | Fig. 3a, 3b |
| 5.5 | `05.5_Dust_Reddening` | reddening, de-reddening, linearised-law control | Fig. 3c, 3d |
| 6 | `06_Paper_Figures` | draws Figs. 1 to 3 from the saved results | Figs. 1 to 3 |

The summary table of the Discussion collects values from notebooks 05.1 to 05.5.

**Training.** Set `SEED` in the parameter cell of `04a_Train_Baseline.ipynb` and run it once per seed, or run all five with papermill:

```bash
for s in 0 1 2 3 4; do
  papermill 04a_Train_Baseline.ipynb runs/04a_seed$s.ipynb -p SEED $s
done
```

**Using the released checkpoints.** To skip training, keep the ten files in `checkpoints/` and the `summary_*.json` files in `results/`, then start from notebook 05.0. Notebooks 01 and 02 still have to be run once to build `data/sdss_tensors.h5`, because the evaluation reads the test spectra from it.

**Figures only.** Notebook 06 reads only the CSV and JSON files in `results/` and the catalogue in `data/`, and runs in seconds. Fig. 1 is TikZ: the notebook compiles it with `pdflatex` when that is installed, and otherwise leaves `figures/fig1_architecture.tex` to compile elsewhere.

## Notes

- **One definition of the models.** Every notebook imports the architectures from `models.py`, and `check_parameter_parity()` stops a notebook if the definition no longer matches the parameter counts of the checkpoints. Layer names must not be changed, or the checkpoints will not load.
- **Deterministic evaluation.** Models are evaluated in inference mode with deterministic cuDNN kernels. Each corrupted tensor is built once and passed through all ten networks, and per-spectrum predictions are cached under `results/step5x/`, so an interrupted notebook resumes where it stopped.
- **Additional analyses.** Some notebooks contain controls and diagnostics that the paper does not report, for example the M2/M3 fills in 05.2, the smoothing and Fourier-shift controls in 05.3, and the line-depth diagnostic in 05.4. Each notebook lists in its header which of its results appear in the paper.
- **Diagnostic figures.** The plots inside notebooks 04a to 05.5 are for inspecting the experiments. The figures of the paper are drawn only by notebook 06.

## Citation

```bibtex
@inproceedings{pham_robustness_2026,
  author    = {Pham, Nhat Khanh and Pham, Minh Tuan},
  title     = {Robustness of Convolutional and Transformer-Based Stellar
               Temperature Estimators to Physical Spectral Artefacts},
  booktitle = {International Symposium on Information and Communication
               Technology (SOICT 2026)},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026},
  note      = {to appear}
}
```

## Acknowledgements

This work uses data from the Sloan Digital Sky Survey V, Data Release 19. Funding for SDSS-V has been provided by the Alfred P. Sloan Foundation, the Heising-Simons Foundation, the National Science Foundation and the Participating Institutions. See [www.sdss.org](https://www.sdss.org).

## Contact

Minh Tuan Pham (corresponding author), pmtuan@dut.udn.vn
Nhat Khanh Pham, nhatkhanh2311@gmail.com

University of Science and Technology, The University of Danang, Vietnam
