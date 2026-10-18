# Machine Learning Assignment 2

**Student:** Jiya Yadav  
**Roll number:** M26CSE004  
**Random seed:** 26004

This repository contains the complete code, fixed data split, configurations, final logs, result tables, figures, and report for the Olivetti faces assignment. Bagging and multiclass SAMME boosting are implemented directly in `src/run_experiments.py`; scikit-learn is used only for the decision-tree weak learner and utility functions.

## Reproduce everything

Create the environment and run the experiment from the repository root:

```bash
conda env create -f environment.yml
conda activate jiya-ml-assignment2
python src/run_experiments.py --repetitions 30 --jobs 8
```

The first run downloads Olivetti faces through `fetch_olivetti_faces`. The saved split is already included in `data/split_seed26004.npz` and `data/split_seed26004.csv`. The program deterministically recreates it from seed 26004 and writes all numerical outputs to `results/`, all plots to `figures/`, and the final run record to `logs/`.

The submitted PDF is `report/Jiya_Yadav_M26CSE004_Assignment2.pdf`. Its source is `report/report.tex`. The report uses the official CVPR 2026 `cvpr.sty` from the CVPR author kit.

## Repository map

- `src/run_experiments.py`: complete experiment and plotting code
- `config/experiment.json`: protocol and hyperparameter grid
- `data/split_seed26004.*`: exact train, validation, and test indices
- `results/`: CSV and JSON result files, including the complete SAMME toy round
- `figures/`: report-ready plots and example faces
- `logs/`: console output and final machine-readable run record
- `report/`: PDF, LaTeX source, and CVPR style file

## Notes

- All model selection uses clean validation accuracy only. Exact ties are resolved toward fewer rounds or estimators and then the simpler setting.
- The masked set is never used for training or tuning.
- Class labels are the integer identity labels returned by scikit-learn, from 0 to 39.
