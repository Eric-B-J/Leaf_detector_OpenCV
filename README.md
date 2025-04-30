# Leaf_detector_OpenCV

🌿 Leaf Detection using OpenCV
This project is a GUI-based leaf classification system built using PyQt5 and core image processing techniques in Python. The goal is to classify leaves into five predefined species based on three categories of features:
Color (LAB Space)

Texture (Gray-Level Analysis)

Shape (Geometric Features)

Users can upload a leaf image and select a feature category for prediction.


📂 Dataset
The dataset used in this project consists of self-collected leaf images, captured manually using a HP scanner. The images were preprocessed to enhance quality and consistency before feature extraction.
Leaves used for this project:
![image](https://github.com/user-attachments/assets/70fabc37-1251-46bf-a4f8-6ee8b4e66d15)
Cleistocalyx operculatus
![image](https://github.com/user-attachments/assets/e2766632-fb97-4ff3-a85a-0dceaa708185)
Cordyline fruticosa
![image](https://github.com/user-attachments/assets/f2c97a6a-1223-4626-b6cf-22eba1613f59)
Pseuderanthemum carruthersii
![image](https://github.com/user-attachments/assets/08f5b9bc-0ec3-483d-9a62-3ba9b6df37be)
Bougainvillea spectabilis
![image](https://github.com/user-attachments/assets/e368bc7a-5911-49e6-a818-9f872768a6bf)
Psidium guajava


🛠 Features
Image preprocessing (resizing, noise reduction, segmentation)

Feature extraction :
Shape descriptors:
Area – total pixel count of the leaf 
Perimeter – boundary length of the leaf shape 
Aspect Ratio – width to height ratio 
Extent – ratio of area to bounding box area 
Solidity – ratio of area to convex hull area 


Color histograms:

a_mean: Average value of the green-red chromatic component
b_std: Variability in the blue-yellow chromatic component


Texture features:

Mean – average gray level 
Variance – spread of intensity values 
Entropy – randomness in pixel distribution
Skewness – asymmetry of intensity distribution 
Kurtosis – peakedness of the distribution 
Contrast – intensity contrast between a pixel and its neighbor



Leaf classification using rule-based and Score-based approaches:
Score- Based Classification:

Compares input features to predefined thresholds for each species.
Each matching feature adds a score to the corresponding species.
Weighted scoring used for texture features.
Shape-based method uses equal weights for all matched features.
Final prediction = Species with the highest total score.
if range_values[0] <= features[feature_name] <= range_values[1]: scores[species] += 1

Rule- Based Classification:
Direct comparison of feature values against fixed thresholds.
● Example: if a_mean >= 127.7: return 'Cordyline fruticosa'



GUI support for user interaction 
![image](https://github.com/user-attachments/assets/5b936743-1987-485d-b02b-0dd8fc107e44)


🧰 Tools & Libraries
Python

OpenCV

Scikit-Image

NumPy

Matplotlib

PyQt5 
