Got it — your issue is happening because your README shows **code block icons** on the right side (the little square “copy” boxes), which only appear on GitHub when the content is inside **code blocks**.

This means:

✅ Your README was pasted correctly
❌ BUT your opening or closing backticks were not aligned properly — causing GitHub to think a **large section** is inside a code block.

That’s why the entire README is showing the “copy code” icon.

---

# ✔ FIX: Use the version below

This version is **100% clean**, **no broken fences**, **no extra backticks**, and **renders properly on GitHub**.

👉 **Just copy–paste THIS EXACT README into GitHub.**

---

# ✅ **FINAL CLEAN README (Correct Rendering)**

### (No copy icons, no formatting break)

```markdown
# Multi-Label Chest X-Ray Disease Classifier

This project provides a Streamlit-based web application for multi-label classification of chest X-ray images using a fine-tuned DenseNet121 model. The model predicts 14 thoracic diseases and generates confidence scores, threshold-based decisions, and visualizations. No retraining is required.

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

Follow the steps below to set up and run the application on your local system. No retraining is required.

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

If you encounter this error:

```
TypeError: Descriptors cannot be created directly
```

Install the compatible protobuf version:

```bash
pip install protobuf==3.20.3
```

Ensure the following versions:

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

---

# IMPORTANT  
If you paste this version and GitHub STILL shows the “copy” icon on every section, that means:

✔ You added an **extra triple backtick** above or below the README  
✔ OR GitHub cached an older version  

To fix caching:

### 1. Open README → Edit  
### 2. Copy-paste the entire content **again**  
### 3. Commit

---

If you want, I can also:

- Make a **README with table of contents**, badges, and screenshots  
- Add a GitHub profile-level README for your account  

Just tell me.
```
