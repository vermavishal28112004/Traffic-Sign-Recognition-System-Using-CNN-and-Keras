# Traffic Sign Recognition System 🚦📷

This project implements a deep learning-based **Traffic Sign Recognition** system using a **Convolutional Neural Network (CNN)** built with **Keras** and **TensorFlow**. It is trained on the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset to accurately classify 43 different types of traffic signs.

---

## 🚗 Project Overview

Traffic sign recognition is essential for intelligent transportation systems and autonomous vehicles. This system uses a supervised learning approach to detect and classify traffic signs from input images with high accuracy.

---

## 📂 Dataset

- **Dataset Used**: German Traffic Sign Recognition Benchmark (GTSRB)  
- **Classes**: 43 different traffic signs  
- **Images**: ~50,000 labeled images in total  
- Includes both training and testing data  

---

## 🧠 Model Architecture

- Multiple convolutional layers (Conv2D) with ReLU activation  
- MaxPooling layers for downsampling  
- Dropout layers to reduce overfitting  
- Fully connected Dense layers for classification  
- Output layer with 43 softmax units (for 43 classes)

---

## ⚙️ Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- OpenCV

---

## 📁 Files Included

```markdown
📦 Traffic-Sign-Recognition-System-Using-CNN-and-Keras  
├── 📁 Documentations  
│   ├── TSRS CNN and Keras PPT.pptx                # Project presentation  
│   ├── TSRS CNN and Keras Report.pdf              # Final project report  
│   ├── TSRS CNN and Keras Research Paper.pdf      # Research paper draft  
│   └── Team Members.txt                           # List of contributors  
│  
└── 📁 traffic_sign_recognition                      
    ├── 📁 Output Screenshot                       # Screenshots of model predictions and results  
    │    └── *.png 
    │  
    ├── 📁 data                                    # Dataset details or link
    |    └── dataset.txt 
    │  
    ├── 📁 notebooks                               # Jupyter notebooks  
    |    ├── 1_Data_Preprocessing.ipynb             # Data cleaning and setup  
    |    ├── 2_Model_Training.ipynb                 # CNN model training  
    |    ├── 3_Model_Evaluation.ipynb               # Evaluation metrics and graphs  
    |    ├── 4_Image_Prediction.ipynb               # Single image prediction  
    |    └── 5_RealTime_Detection.ipynb             # Real-time webcam detection                              
    │
    └── 📁 testIMG                                 # Test images for inference 
         └── *.png
```

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/tarangver/Traffic-Sign-Recognition-System-Using-CNN-and-Keras.git
   cd Traffic-Sign-Recognition-System-Using-CNN-and-Keras
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebooks step-by-step from the `notebooks/` folder:
    ```bash
       1_Data_Preprocessing.ipynb
       2_Model_Training.ipynb
       3_Model_Evaluation.ipynb
       4_Image_Prediction.ipynb
       5_RealTime_Detection.ipynb
    ```
---

## 📈 Model Performance

- Achieved high classification accuracy (>95%) on validation and test sets  
- Evaluation includes accuracy, loss graphs, and confusion matrix  

---

## 🧪 Key Features

- Image preprocessing and normalization  
- CNN model with dropout for regularization  
- Jupyter-based training, validation, and testing pipeline  
- Real-time prediction using webcam  
- Supports single image and batch inference

---

## 🌟 Applications

- Autonomous vehicles (ADAS)  
- Smart traffic management systems  
- Road safety and sign compliance monitoring  
- AI-powered driving assistance tools

---

## 🙋‍♂️ Author & C0-Author

**Tarang Verma**  
GitHub: [@tarangver](https://github.com/tarangver)  
LinkedIn: [@verma-tarang](https://www.linkedin.com/in/verma-tarang/)

**Vishal Verma**  
GitHub: [@vermavishal28112004](https://github.com/vermavishal28112004)  
LinkedIn: [@vishal-verma-14796b286](https://www.linkedin.com/in/vishal-verma-14796b286/)

---

## 📄 License

This project is licensed under the **MIT License**.  
Free to use for personal, academic, or commercial purposes with proper attribution.
