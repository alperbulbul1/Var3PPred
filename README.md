[![DOI](https://zenodo.org/badge/652651218.svg)](https://zenodo.org/doi/10.5281/zenodo.10324912)
# Var3PPred

Var3PPred: Variant prediction based on 3-D structure and sequence analyses of protein-protein interactions on Autoinflammatory Disease

Var3PPred is a pathogenicity classifier designed to identify pathogenic variants in genes associated with autoinflammatory disorders. This comprehensive approach integrates protein-protein interaction analysis and 3D structural information.

## Dataset
- **Source**: Infevers database
- **Variants**: 702 missense disease-associated variants from 35 genes linked to Systemic Autoinflammatory Diseases (SAIDs)
- **Balancing**: SMOTE algorithm used to balance dataset (130 benign and 572 pathogenic variations)

## Methodology
- **Protein-Protein Interaction Analysis**: Utilized STRING and Intact databases for 3D docking analysis
- **Feature Integration**: Integrated sequential and structural features including ∆∆G, accessible surface area, volume, pLDDT scores, and PSIC scores
- **Feature Calculation**: Features calculated using PyRosetta and AF2 computed structures

## Machine Learning
- **Algorithms**: Six algorithms tested, with random forest classifier found to be most effective
- **Performance**: Random forest achieved AUROC of 99% on test set, outperforming SIFT, PolyPhen, and CADD on an unseen test set of a SAID-related gene

The Merge-Last-vus.csv file contains descriptions of Variants of Unknown Significance (VUS) along with their Var3PPred predictions in the last column (Prediction(0-1), **0**: Benign, **1**: Pathogenic). VEP annotations for these variants are provided in the VEP-Annotation-variants.csv file. The Model_Training.py script contains training methods and SHAP analysis scripts, while the ZDOCK_and_SPRINT_weighted-scores.py script assigns weights to ZDOCK and SPRINT scores based on the HGPEC rank score of genes for systemic autoinflammatory diseases. The pairwised_decision_boundaries_of_models.py file contains scripts for analyzing pairwise relationships of features in the six selected machine learning models with decision boundaries.
