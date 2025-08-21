Overview

Medical imaging, particularly MRI and histopathological scans, is a cornerstone of diagnosing brain, neuro, and spine disorders. Traditional manual assessment is labor-intensive, subjective, and prone to inconsistencies, making automated and precise classification crucial for clinical decision-making. This project presents a comparative deep learning approach for automated diagnosis using:

ResNet
Bidirectional Encoder representation from Image Transformers (BEiT)
To enhance performance, Bio-Inspired Optimization (Whale Optimization Algorithm - WOA) and Grid Search were integrated into the pipeline for hyperparameter tuning.

Objectives

Automate the classification of brain, neuro, and spine disorders from MRI images.
Compare the efficiency of ResNet vs BEiT for medical image classification.
Employ Bio-Inspired Optimizers (WOA) to refine model hyperparameters.
Achieve higher diagnostic accuracy while minimizing training inefficiencies.

Dataset
Link: https://data.mendeley.com/datasets/d73rs38yk6/1

Source: Benchmark Diagnostic MRI and Medical Imaging Dataset

Size: 34,192 MRI images

Classes: 40 clinically relevant categories

Type: Brain, Neuro, and Spine disorder MRI scans

Methodology

Data Preprocessing
Normalization & resizing of MRI images
Train/Validation/Test splits

Models Used
ResNet (Convolutional Backbone)
BEiT (Transformer-based pre-trained model)

Optimization Strategy
Grid Search for systematic hyperparameter tuning
Whale Optimization Algorithm (WOA) for bio-inspired metaheuristic optimization

Evaluation Metrics
Precision, Recall, F1-score
Confusion Matrix

Results

ResNet → 91.76% Accuracy
BEiT → 83.56% Accuracy

ResNet outperformed BEiT in diagnostic precision, but BEiT showed potential in efficiency and transformer-based adaptability.

Tech Stack

Programming Language: Python
Deep Learning Frameworks: PyTorch / TensorFlow
Libraries:torch, torchvision, transformers (modeling), scikit-learn (metrics & evaluation), numpy, pandas (data handling), matplotlib, seaborn (visualizations)

Future Work

Extend to multi-modal learning (MRI + clinical reports).
Apply other bio-inspired algorithms (PSO, GA, ACO) for optimization.
Integrate explainability (Grad-CAM, SHAP) for medical interpretability.
Build a web-based diagnostic tool for clinical use.

Citation

If you use this work in your research, please cite:

@project{Comparative-Study-ResNet-BEiT-with-Bio-Inspired-Optimizers-for-Neuro-Disorder-Classification,
  title={Comparative Study with ResNet and BEiT with Bio-Inspired Optimizers for Brain, Neuro, and Spine Disorders Classification},
  author={Samiksha Sandeep Zokande},
  year={2025},
  note={Comparative-Study-ResNet-BEiT-with-Bio-Inspired-Optimizers-for-Neuro-Disorder-Classification}
}

Contributing

Contributions are welcome! Fork the repo, make changes, and submit a PR.

Contact

For queries or collaborations, reach out at:
samikshazokande29@gmail.com
