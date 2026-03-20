# Multimodal Transformer for CT Imaging and EHR Risk Prediction

## Overview

This project explores the development of a multimodal deep learning framework for clinical risk prediction by jointly leveraging medical imaging and structured Electronic Health Record (EHR) data. The work investigates how computed tomography (CT) imaging features can be combined with patient clinical variables to improve predictive modeling of medical outcomes.

The proposed approach utilizes transformer-based architectures to enable feature fusion between radiological image embeddings and tabular EHR data, allowing the model to capture complementary patterns across modalities. By integrating computer vision and clinical data modeling techniques, this project aims to evaluate the potential of multimodal learning for enhanced healthcare risk prediction.

## Problem Statement

Clinical risk prediction models typically rely on structured EHR variables such as laboratory results, demographics, and comorbidities. While these features provide valuable clinical signals, they do not capture information contained in medical imaging data.

CT scans provide high-resolution anatomical information that may reveal patterns associated with disease severity and patient outcomes. However, combining imaging data with tabular clinical data is challenging due to differences in representation and feature structure.

This project investigates whether multimodal deep learning models can improve clinical risk prediction by integrating CT imaging features with structured EHR data using transformer-based feature fusion.

## Proposed Approach

The proposed system combines two primary data modalities:

1. CT Imaging (Computer Vision)

- Feature extraction from CT scan images
- Generation of radiological image embeddings

2. Structured EHR Data

- Demographics
- Laboratory measurements
- Clinical variables

These two representations are integrated using a transformer-based fusion architecture to learn joint representations for risk prediction.
