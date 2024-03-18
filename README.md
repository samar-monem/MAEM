#  MAEM : synergyistic drug combinatioon deep learning model 
Source code for the paper: A Multi-View Feature Representation for Predicting Drugs Combination Synergy based on Ensemble and Multi-Task Attention Models
# Data
The datasets used in the paper is the O'neil benchmark dataset.
# Components

    features_data/drug_finger - the fingerprint features for drugs 
    features_data/drug_target - the target features for drugs 
    features_data/redkit_drug - the mordred features for drugs 
    features_data/views0b123 - the multi-view graph features for drug
    features_data/vae_copy256 - the copy number features for cancer cell line 
    features_data/vae_expression256 - the gene expression features for cancer cell line 
    features_data/vae_mutation256 - the gene mutation features for cancer cell line 
    features_data/vae_proteomics256 - the gene proteomics features for cancer cell line 
    results/synergy_class - results for sixteen prediction output values for predicting the synergy class label
    results/synergy_score - results for sixteen prediction output for predicting the synergy score value
    train/ help_functions - the normalized and repeated function to pocess data for training
    train/ model - the implementation of the MAEM model
    train/ rub - run code example
   
 # Usage
     The 'train/ help_functions' contain the processing functions to input data to the 'train/ model'  MAEM implementation. 
     The  'train/ rub' shows an examples of running code. first it applien the processing data for each view of drug and cancer cell line. Then, the model is trained and can        predict new drug combinations. 
# Dependencies
    xlrd==2.0.1 
    Pandas==1.3.5
    tensorflow == 2.13.0
    numpy==1.24.3
 
