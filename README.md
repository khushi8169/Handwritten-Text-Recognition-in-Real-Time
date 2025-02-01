
## **Project Title: Real-Time Handwritten Text Recognition (Hindi & English)**  

### **Overview**
This project is a **real-time handwritten text recognition system** that captures an image from a webcam and predicts handwritten text in **Hindi (Devanagari) and English** using deep learning models. The predictions are then saved into a Word document.

### **Features**
- Captures **handwritten input** using a webcam.  
- Recognizes **English sentences** and **Hindi characters** using trained CNN and LSTM-based models.  
- Saves predictions in a **Word document** for easy access.  
- Uses **OpenCV for image capture** and **TensorFlow for deep learning inference**.  

---

## **Installation and Setup**
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/handwritten-text-recognition.git
cd handwritten-text-recognition
```

### **2. Install Dependencies**
Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```
(Note: Add the necessary dependencies in `requirements.txt`)

### **3. Download or Train Models**
- Place your trained **Hindi model (`devanagari_model.keras`)** and **English model (`iam_model.keras`)** in the project directory.  
- If models are missing, you can train them using `train.py`.

### **4. Run the Application**
To capture an image and perform predictions:
```bash
python predict.py
```
---

## **Project Structure**
```
📂 handwritten-text-recognition
│-- 📜 camera.py        # Captures images from the webcam
│-- 📜 predict.py       # Loads models & predicts handwritten text
│-- 📜 preprocess.py    # Preprocesses images for inference
│-- 📜 train.py         # Trains CNN+LSTM models for text recognition
│-- 📜 requirements.txt # Required dependencies
│-- 📜 README.md        # Project documentation
```
---

## **How It Works**
1. The script captures an image using **`camera.py`**.
2. The **`predict.py`** script preprocesses the image and loads trained models.  
3. Hindi characters are predicted using the **Hindi CNN model (`devanagari_model.keras`)**.  
4. English sentences are predicted using the **English LSTM model (`iam_model.keras`)**.  
5. The output is saved in a **Word document (`output.docx`)**.  

---

## **Tech Stack**
- **Programming Language**: Python  
- **Libraries**: OpenCV, TensorFlow/Keras, NumPy, Docx  
- **Deep Learning Architecture**: CNN + LSTM  

---

## **Future Improvements**
- Improve **accuracy** using **data augmentation**.  
- Implement **real-time continuous recognition** instead of single-frame capture.  
- Develop a **web interface** for better user experience.  

---
