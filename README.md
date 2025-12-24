How to Run This Project After Cloning

Follow the steps below to set up and run the Multi-Label Chest X-Ray Disease Classifier on your local system. This project does not require retraining the model.

**1. Clone the Repository**

git clone https://github.com/MdAshrafhussain889/chest-xray-disease-classifier.git
cd chest-xray-disease-classifier


**2. Create a Conda Environment**

conda create -n chestxray-env python=3.10 -y
conda activate chestxray-env


**3. Install Dependencies**

pip install --upgrade pip
pip install -r requirements.txt



**4.Navigate to the Streamlit Application**

cd streamlit_app




**5. Run the Application**


streamlit run app.py


The application will open at:


http://localhost:8501


**6. Upload and Analyze X-Ray Images**

 Supported formats: PNG, JPG, JPEG
 The model outputs:

⦁	Predicted disease labels
⦁	Confidence scores
⦁	Threshold-based YES/NO decisions
⦁	Score vs. threshold comparison chart
⦁	Detailed results table

