# Project Documentation

## Overview
This project focuses on analyzing error correlations across machine learning models.

### Data Directory
- **/data**: This directory is intended for storing datasets used in the project. 
- **/llm_main**: The functions for finetuning LLMs.
- **/plots**: This directory is intended for storing correlation plots of image and tabular models.
- **/test_loss**: This directory is intended for storing errors of image and tabular models.
- **/result**: This directory is intended for storing correlation plots and errors of LLMs.

### Jupyter Notebooks
- **training_pipeline_LLM.ipynb**: The core notebook for fine-tuning LLMs. It includes steps for data loading, preprocessing, model training, and testing.
- **accumulation_image.ipynb**: The notebook for obtaining correlations of errors in image classification models.
- **accumulation_LLM.ipynb**:The notebook for obtaining correlations of errors in LLMs.
- **accumulation_tab.ipynb**: The notebook for obtaining correlations of errors in regression model.


