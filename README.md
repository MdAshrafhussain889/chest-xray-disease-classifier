

```markdown
# Multi-Label Chest X-Ray Disease Classifier

This project provides a Streamlit-based web application for multi-label classification of chest X-ray images using a fine-tuned DenseNet121 model. The model predicts 14 thoracic diseases and generates confidence scores, threshold-based decisions, and visualizations. No retraining is required.

---

## Overview

This application analyses a chest X-ray image and predicts the likelihood of multiple diseases simultaneously. It uses a DenseNet121 architecture fine-tuned on the NIH ChestXRay14 dataset. The interface is built with Streamlit and runs entirely on the user's local machine.

---

## Supported Disease Classes

The model predicts the following 14 chest diseases:

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
│
├── final_densenet121_model.h5
├── model_metadata.json
├── optimal_thresholds.npy
├── class_weights.npy
│
├── requirements.txt
└── README.md

````

---

## Requirements

The project uses the following core dependencies (included in `requirements.txt`):

- tensorflow==2.10.0  
- numpy==1.23.5  
- protobuf==3.20.3  
- streamlit>=1.25.0  
- opencv-python  
- pillow  
- matplotlib  
- pandas  

These versions ensure compatibility with TensorFlow 2.10 and prevent protobuf-related errors.

---

## How to Run This Project After Cloning

Follow the steps below to set up and run the Multi-Label Chest X-Ray Disease Classifier on your local system. This project does not require retraining the model.

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

### 6. Upload and Analyze X-Ray Images

Supported image formats: PNG, JPG, JPEG

The model will generate:

* Predicted disease labels
* Confidence scores
* Threshold-based YES/NO decisions
* Score vs. threshold comparison chart
* Detailed results table

---

## Troubleshooting

If you encounter the following error:

```
TypeError: Descriptors cannot be created directly
```

Install the correct protobuf version:

```bash
pip install protobuf==3.20.3
```

Ensure that:

* Python version is 3.10
* TensorFlow version is 2.10.0
* NumPy version is 1.23.5
* Protobuf version is 3.20.3

---

## License

This project is intended for academic and research purposes only. It is not a medical diagnostic tool.

---

## Author

Md Ashraf Hussain
Artificial Intelligence and Machine Learning

```

---

If you want, I can also create:

- A shorter README  
- A more detailed research-style README  
- A professional cover image/banner for GitHub  

Just tell me.
```
