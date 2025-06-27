# 🩺 Pneumonia Detection using CNN (PyTorch)

A deep learning-based AI system that detects pneumonia from chest X-ray images using a custom-built Convolutional Neural Network (CNN) in PyTorch.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project aims to detect **Pneumonia** from chest X-ray images by training a CNN classifier on a publicly available dataset. It performs:
- Image classification (`NORMAL` vs `PNEUMONIA`)
- Model evaluation using accuracy and confusion matrix
- Single image prediction via script
- (Optional) Future upgrade to web or cloud app

---

## 🧠 Technologies Used

- 🔶 Python
- 🔶 PyTorch
- 🔶 Torchvision
- 🔶 scikit-learn
- 🔶 PIL (Pillow)
- 🔶 VS Code

---

## 📦 Dataset

**Chest X-Ray Images (Pneumonia)**  
🧪 Source: [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

> Note: The dataset is not uploaded to GitHub due to size limits.  
> Please download it manually from Kaggle and extract it into a folder named `chest_xray/`.

---

## 📁 Folder Structure

pneumonia-detector/
├── chest_xray/ # Dataset folder (downloaded from Kaggle)
│ ├── train/
│ ├── test/
│ └── val/
├── cnn_model.py # CNN model training & evaluation script
├── predict.py # Single image prediction script
├── saved_model.pth # Trained model file
├── requirements.txt # Required Python libraries
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🚀 How to Run the Project

 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/pneumonia-detection.git
cd pneumonia-detection
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Train the CNN Model
bash
Copy
Edit
python cnn_model.py
This will train the model for 5 epochs and save it as saved_model.pth.

4. Predict on a New Image
Edit the image path inside predict.py:

python
Copy
Edit
predict_image("chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg")
Then run:

bash
Copy
Edit
python predict.py
✅ Results
✅ Test Accuracy: ~94–96%

✅ Confusion Matrix generated after training

✅ Predicts single image with output: PNEUMONIA or NORMAL

📌 Future Enhancements
🔹 Flask web app for live prediction in browser

🔹 Gradio or Streamlit deployment for public demo

🔹 Integration of Transfer Learning with ResNet/VGG

👤 Author
Arya Rakshe
📧 [coolrxkshe69@gmail.com]
🌐 github.com/Coolrxkshe

📄 License
This project is licensed under the MIT License.
Feel free to use, modify, and share it with attribution.
