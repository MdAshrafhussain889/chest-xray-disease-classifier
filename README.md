

```markdown
# Multi-Label Chest X-Ray Disease Classifier

This project provides a Streamlit-based web application for multi-label classification of chest X-ray images using a fine-tuned DenseNet121 model. The system predicts 14 thoracic diseases and outputs confidence scores, threshold-based YES/NO decisions, and visualizations. The model is already trained, and no retraining is required.

---

## Overview

This application analyzes a chest X-ray image and predicts the likelihood of multiple diseases simultaneously. It uses a DenseNet121 architecture fine-tuned on the NIH ChestXRay14 dataset. The interface is built with Streamlit and runs entirely on the user's local machine.

---

## Supported Disease Classes

The model predicts the following 14 thoracic diseases:

- Atelectasis  
- Consolidation  
- Infiltration  
- Pneumothorax  
- Edema  
- Emphysema  
- Fibrosis  
- Effusion  
- Pneumonia  
- Pleural Thickening  
- Cardiomegaly  
- Nodule  
- Mass  
- Hernia  

---

## Project Structure

```

chest-xray-disease-classifier/
│
├── streamlit_app/
│   ├── app.py
│   ├── model_loader.py
│   ├── pdf_generator.py
│   ├── final_densenet121_model.h5
│   ├── model_metadata.json
│   ├── optimal_thresholds.npy
│   └── uploads/
│
├── phase1_best_model.h5
├── phase2_best_model.h5
├── model_metadata.json
├── class_weights.npy
├── optimal_thresholds.npy
├── temperature_scaling.json
├── test_results.csv
├── project.ipynb
├── training_and_results.png
├── phase1_training_history.png
├── phase2_training_history.png
├── requirements.txt
└── README.md

````

---

## Requirements

The following dependencies are included in `requirements.txt`:

- tensorflow==2.10.0  
- numpy==1.23.5  
- protobuf==3.20.3  
- streamlit>=1.25.0  
- opencv-python  
- pillow  
- matplotlib  
- pandas  

These versions ensure compatibility with TensorFlow 2.10 and avoid protobuf-related errors.

---

## How to Run This Project After Cloning

Follow the steps below to set up and run the application on your local system.

### 1. Clone the Repository

```bash
git clone https://github.com/MdAshrafhussain889/chest-xray-disease-classifier.git
cd chest-xray-disease-classifier
````

### 2. Create a Conda Environment

```bash
conda create -n chestxray-env python=3.10 -y
conda activate chestxray-env
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Navigate to the Streamlit Application

```bash
cd streamlit_app
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## Upload and Analyze X-Ray Images

Supported file formats:

* PNG
* JPG
* JPEG

The model outputs:

* Predicted disease labels
* Confidence scores
* Threshold-based YES/NO decisions
* Score vs. threshold comparison chart
* Detailed results table

---

## Troubleshooting

If you encounter the error below:

```
TypeError: Descriptors cannot be created directly
```

Install the compatible protobuf version:

```bash
pip install protobuf==3.20.3
```

Ensure the following versions are used:

* Python 3.10
* TensorFlow 2.10.0
* NumPy 1.23.5
* Protobuf 3.20.3

---

## License

This project is intended for academic and research purposes only. It is not a medical diagnostic tool.

---

## Author

**Md Ashraf Hussain**
Artificial Intelligence and Machine Learning

```


