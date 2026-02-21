# Cancer-detection
Domain shift caused by variations in staining protocols, scanner hardware, and
tissue preparation across institutions remains a key obstacle to deploying deep learning in
computational pathology. We propose a domain-invariant framework for binary cancer
classification from histopathological images, integrating a Vision Transformer (ViT-B/16)
backbone with an adversarial domain training objective via a gradient reversal layer. A multi-stage
data curation pipeline—comprising perceptual hash-based deduplication, rule-based quality
filtering, and ensemble-driven label verification—is applied to 29,364 H&E-stained patches from
two publicly available repositories with distinct tissue origins and imaging characteristics:
LC25000 and NCT-CRC-HE-100K. The model achieves 99.83% accuracy and an AUROC of
0.9995 on the held-out test set, while 10-fold cross-validation confirms exceptional stability (mean
accuracy 99.60% ± 0.24%). Grad-CAM analysis further shows that predictions are grounded in
clinically meaningful morphology rather than domain-correlated artifacts, confirming that the
adversarial objective effectively suppresses source-specific feature encoding. These results
establish the proposed framework as a robust and interpretable approach to mitigating domain shift
toward reliable AI-assisted diagnosis in heterogeneous clinical settings.
