# Reproduction of results from "The Virtue of Complexity" (VoC) paper

==============================

This project reproduces main empirical findings from Kelly, Malamud and Zhou, 2023, "The virtue of complexity in return prediction", Journal of Finance.

Project Organization
------------

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
                              generated with `conda list -e > requirements.txt`
    └── src                <- Run them from terminal
        │
        ├── features       <- make RFFs
        │
        ├── models         <- ridge_solvers                    
        │
        ├── utils          <- unpack_results
        │
        └── visualization  <- Scripts to create main VoC figures 


--------
