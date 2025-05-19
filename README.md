#  Building Energy Efficiency Classification using Neural Networks
---
##  Overview
This project uses machine learning models—specifically a neural network (MLPClassifier) and Logistic Regression—to classify buildings as energy efficient or not. The features include architectural and design parameters from real-world building data. The goal is to compare performance between traditional and neural approaches.

---

##  Dataset
- **Source:** [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)
- **Samples:** 768 buildings
- **Features:** 
  - Relative Compactness
  - Surface Area
  - Wall Area
  - Roof Area
  - Overall Height
  - Orientation
  - Glazing Area
  - Glazing Area Distribution
- **Target:** Binary classification (Efficient = 1 if Heating Load < 15)

---

##  Models Used
- **MLPClassifier (Multilayer Perceptron)** with 2 hidden layers
- **Logistic Regression** (baseline for comparison)

---

##  Results
| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | ~93.5%   |
| MLPClassifier        | ~99.3%   |

- Neural network outperforms traditional model in accuracy and recall.

---

## Key Takeaways
-	Scaling inputs is crucial for neural networks.
-	MLPClassifier generalizes well with fewer neurons when tuned right.
-	Logistic Regression is still a reliable baseline for tabular data.

## Future Improvements
-	Use cross-validation for more robust evaluation
-	Add more model types (e.g. Random Forest, XGBoost)
-	Convert into a Streamlit web app for real-time predictions

---

🙋‍♂️ Author

**Varun Chaturvedi**
[LinkedIn](https://www.linkedin.com/in/varunchaturvedii/)
