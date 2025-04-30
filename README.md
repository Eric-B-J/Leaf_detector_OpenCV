

# 🌿 Leaf Detector using OpenCV

This project is a **GUI-based leaf classification system** built using **PyQt5** and core image processing techniques in Python. The goal is to classify leaves into **five predefined species** based on three categories of features:

- **Color** (in LAB color space)  
- **Texture** (Gray-Level analysis)  
- **Shape** (Geometric features)

Users can upload a leaf image and select a feature category for prediction through a user-friendly GUI.

---

## 📂 Dataset

The dataset consists of **self-collected leaf images**, scanned manually using an **HP scanner** to ensure high resolution and consistency. The images were preprocessed to improve quality before feature extraction.

### Leaves used in this project:

#### 1. *Cleistocalyx operculatus*  
![Cleistocalyx operculatus02](https://github.com/user-attachments/assets/5f02fcde-0ba7-46eb-9075-1d68dfb3b1ad)

#### 2. *Cordyline fruticosa*  
![Cordyline_fruticosa_01](https://github.com/user-attachments/assets/641f81c1-13d1-4ad3-964a-032cbafae100)

#### 3. *Pseuderanthemum carruthersii*  
![Psuderanthemum_carruthersii_02](https://github.com/user-attachments/assets/7ef562ba-7655-47af-b9b3-ff435e99681c)


#### 4. *Bougainvillea spectabilis*  
![Bougainvillea_01](https://github.com/user-attachments/assets/7cd1ae2f-ee6b-4906-b0fc-29b622a667b2)

#### 5. *Psidium guajava*  
![Psidium gauvaja_02](https://github.com/user-attachments/assets/2d0b14e4-e2ae-47c5-a6a7-9df0680e7dfd)


---

## 🛠 Features

### 🔹 Image Preprocessing

- Resizing  
- Noise Reduction  
- Segmentation  

### 🔹 Feature Extraction

**Shape Descriptors:**

- **Area** – Total pixel count of the leaf  
- **Perimeter** – Boundary length of the leaf shape  
- **Aspect Ratio** – Width to height ratio  
- **Extent** – Ratio of area to bounding box area  
- **Solidity** – Ratio of area to convex hull area  

**Color Histograms (LAB space):**

- `a_mean` – Average of green-red chromatic component  
- `b_std` – Variability in the blue-yellow chromatic component  

**Texture Features:**

- **Mean** – Average gray level  
- **Variance** – Spread of intensity values  
- **Entropy** – Randomness in pixel distribution  
- **Skewness** – Asymmetry of intensity distribution  
- **Kurtosis** – Peakedness of the distribution  
- **Contrast** – Intensity contrast between a pixel and its neighbor  

---

## 🧠 Classification Methods

### ✅ Score-Based Classification

- Compares input features to thresholds per species  
- Each matched feature adds a score  
- Weighted scoring for texture features  
- Shape-based method uses equal weights  
- **Final prediction** = species with **highest total score**

```python
if range_values[0] <= features[feature_name] <= range_values[1]:
    scores[species] += 1
```

### ✅ Rule-Based Classification

- Uses direct feature-to-threshold mapping  
- Example rule:

```python
if a_mean >= 127.7:
    return 'Cordyline fruticosa'
```

---

## 💻 GUI Interface

A simple and interactive GUI built with **PyQt5** allows users to:

- Upload a leaf image  
- Select feature type (color, texture, shape)  
- View predicted species  

### Screenshot:
![GUI](https://github.com/user-attachments/assets/5b936743-1987-485d-b02b-0dd8fc107e44)

---

## 🧰 Tools & Libraries

- Python  
- OpenCV  
- Scikit-Image  
- NumPy  
- Matplotlib  
- PyQt5  

---
