# Reproduction of results from "The Virtue of Complexity" (VoC) paper

==============================

This project reproduces main empirical findings from Kelly, Malamud and Zhou, 2023, "The virtue of complexity in return prediction", Journal of Finance.

Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for using this project.
    ├── data
    │   ├── external       <- Data from extending the paper reproduction, e.g. additional variables.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Project documentation, notes, manuals, and all other explanatory materials.
    │    ├── papers        <- Papers used for the project.
    │    └── notes         <- Notes on the project, data, and code. 
    │
    ├── models             <- The main script for fitting models for different seeds used for making RFFs
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `01-jqp-initial-data-exploration`.
    │    
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    |   ├── main           <- The final report.
    |   ├── presentation   <- Presentation slides and notes.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── setup.py           <- makes project pip installable (pip install -e .) so src can be imported

--------
