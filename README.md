# 🏨 Travel & Hospitality Reinforcement Learning for Dynamic Pricing

## 📌 Project Overview

This project focuses on **hotel demand prediction and dynamic pricing using Machine Learning and Markov Decision Processes (MDP)**.

The objective is to analyze historical hotel booking data, predict booking cancellations/demand patterns, and formulate a dynamic pricing strategy based on the predicted demand.

The project combines **Data Preprocessing, Exploratory Data Analysis (EDA), Feature Engineering, SMOTE, Machine Learning, and Reinforcement Learning concepts** to support data-driven pricing decisions in the hospitality industry.

---

## 🎯 Objectives

* Analyze historical hotel booking data.
* Understand important factors affecting hotel bookings and cancellations.
* Perform data cleaning and preprocessing.
* Create meaningful features through feature engineering.
* Handle class imbalance using **SMOTE**.
* Train and compare Machine Learning models.
* Predict hotel booking cancellations/demand.
* Formulate a **Markov Decision Process (MDP)** for dynamic pricing.
* Define pricing actions based on hotel demand.
* Calculate rewards for different pricing decisions.
* Develop a data-driven dynamic pricing strategy.

---

## 📊 Dataset

### Hotel Booking Demand Dataset

The project uses the **Hotel Booking Demand Dataset**, containing booking information for city hotels and resort hotels.

### Dataset Details

* **Records:** 119,390
* **Original Features:** 32
* **Target Variable:** `is_canceled`

### Target Distribution

| Booking Status    |  Count |
| ----------------- | -----: |
| Not Cancelled (0) | 75,166 |
| Cancelled (1)     | 44,224 |

The target variable indicates whether a booking was canceled.

* `0` → Booking not canceled
* `1` → Booking canceled

---

## 🔄 Project Workflow

```text
Hotel Booking Dataset
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Encoding & Scaling
        ↓
Train-Test Split
        ↓
SMOTE
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Demand Analysis
        ↓
MDP Formulation
        ↓
Pricing Actions
        ↓
Reward Calculation
        ↓
Dynamic Pricing Strategy
```

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Removed unnecessary columns.
* Handled missing values.
* Filled missing `country` values using the mode.
* Filled missing `children` values using the median.
* Handled missing `agent` and `company` values.
* Removed `reservation_status` and `reservation_status_date` where required to avoid target leakage.
* Converted categorical variables into numerical representations.
* Applied feature scaling where required.

After preprocessing, the feature dataset contained approximately **33 features**.

---

## 🔧 Feature Engineering

Feature engineering was performed to improve the representation of the hotel booking data.

Examples of useful features include:

* Total stay duration
* Total number of guests
* Total special requests
* Total previous bookings
* Booking lead time
* Room-related information
* Customer-related information
* Deposit and booking-related characteristics

Feature engineering helps the models identify relationships between booking characteristics and cancellation/demand patterns.

---

## 📈 Exploratory Data Analysis

EDA was performed to understand the structure and patterns in the dataset.

The analysis included:

* Distribution of booking cancellations
* Hotel type analysis
* Lead-time analysis
* Customer type analysis
* Deposit type analysis
* Monthly booking trends
* Country-wise booking patterns
* Numerical feature distributions
* Correlation analysis

### Correlation Heatmap

A correlation heatmap was used to identify relationships between numerical variables and detect highly correlated features.

---

## ⚖️ Handling Class Imbalance with SMOTE

The original dataset contains an imbalance between canceled and non-canceled bookings.

**SMOTE (Synthetic Minority Oversampling Technique)** was applied after the train-test split to balance the training data.

Example:

```text
Before SMOTE:
Class 0 → 60,133
Class 1 → 44,??? 
```

After SMOTE:

```text
Class 0 → 60,133
Class 1 → 60,133

Training Shape:
(120266, 33)
```

SMOTE was applied **only to the training data** to prevent information leakage into the test set.

---

## 🤖 Machine Learning Models

The project uses supervised Machine Learning models for hotel booking prediction.

### Models Used

1. **Random Forest Classifier**
2. **LightGBM Classifier**

### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to produce a robust prediction.

Advantages:

* Handles nonlinear relationships.
* Works well with mixed feature types.
* Reduces overfitting compared with a single decision tree.
* Provides feature importance.

### LightGBM

LightGBM is a gradient boosting framework designed for efficient and high-performance Machine Learning.

Advantages:

* Fast training.
* Efficient with large datasets.
* Handles complex nonlinear relationships.
* Provides strong classification performance.

LightGBM was selected as the primary model based on its overall performance.

---

## 📊 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Training Time

Example evaluation results:

| Metric   | Random Forest | LightGBM |
| -------- | ------------: | -------: |
| Recall   |        0.9024 |   0.8843 |
| F1 Score |        0.9143 |   0.8902 |
| ROC-AUC  |        0.9747 |   0.9640 |

The final model selection should consider the project objective and the balance between false positives and false negatives.

---

# 🧠 Markov Decision Process (MDP)

After the Machine Learning stage, an **MDP-based dynamic pricing framework** is formulated.

The MDP allows the pricing system to select an appropriate pricing action based on the current hotel demand state.

### MDP Components

An MDP consists of:

```text
MDP = (S, A, P, R, γ)
```

Where:

* **S** → Set of states
* **A** → Set of actions
* **P** → Transition probabilities
* **R** → Reward function
* **γ** → Discount factor

---

## 🏨 State

The hotel demand state represents the current level of demand.

Example states:

```text
Low
Medium
High
```

The state can be determined using predicted demand or booking-related information.

Example:

```text
Predicted Demand
       ↓
 ┌─────┼─────┐
 ↓     ↓     ↓
Low  Medium  High
```

---

## 💰 Actions

The pricing agent can select one of three actions:

| Action | Description    |
| ------ | -------------- |
| 0      | Decrease Price |
| 1      | Keep Price     |
| 2      | Increase Price |

The purpose is to adjust the room price according to the predicted demand.

### Example

```text
Low Demand    → Decrease Price
Medium Demand → Keep Price
High Demand   → Increase Price
```

---

## 🎁 Reward

The reward represents the benefit obtained from selecting a particular pricing action.

A suitable reward structure can encourage:

* Higher occupancy
* Higher revenue
* Appropriate pricing
* Reduced cancellation risk
* Better utilization during low-demand periods

Example:

```text
High Demand + Increase Price → Positive Reward
Medium Demand + Keep Price   → Positive Reward
Low Demand + Decrease Price  → Positive Reward
```

The reward function can be further customized using actual hotel revenue and occupancy information.

---

## 🔄 MDP Example

```text
Current State
     ↓
  Medium
     ↓
Choose Action
 ┌────┼────┐
 ↓    ↓    ↓
-Price Keep +Price
 └────┼────┘
      ↓
 Calculate Reward
      ↓
Next State
```

The agent evaluates the available actions and selects the action that provides the highest expected reward.

---

## 📌 Sample MDP Output

The current implementation uses three pricing actions:

```text
Actions:
0 : Decrease Price
1 : Keep Price
2 : Increase Price
```

Example output:

```text
Average Reward: 0.8396459259569479
```

This indicates that the implemented pricing policy is generating a positive average reward under the defined reward function.

---

## 🗂️ Project Structure

```text
Infotact-DS-ML-Project-2-Travel-and-Hospitality-Reinforcement-Learning-for-Dynamic-Pricing/
│
├── data/
│   └── hotel_bookings.csv
│
├── processed_data/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── models/
│   └── lightgbm_model.pkl
│
├── notebooks/
│   └── analysis.ipynb
│
├── data_preprocessing.py
├── eda.py
├── mdp.py
├── model_training.py
├── requirements.txt
├── README.md
└── .gitignore
```

*Adjust the filenames above to match the exact files currently in your repository.*

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Imbalanced-learn**
* **LightGBM**
* **Joblib**
* **Reinforcement Learning / MDP**
* **Git & GitHub**
* **VS Code**

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Sherinjohn6/Infotact-DS-ML-Project-2-Travel-and-Hospitality-Reinforcement-Learning-for-Dynamic-Pricing.git
```

Move into the project directory:

```bash
cd Infotact-DS-ML-Project-2-Travel-and-Hospitality-Reinforcement-Learning-for-Dynamic-Pricing
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1. Data Preprocessing

```bash
python data_preprocessing.py
```

### 2. Exploratory Data Analysis

```bash
python eda.py
```

### 3. Model Training

```bash
python model_training.py
```

### 4. MDP Dynamic Pricing

```bash
python mdp.py
```

---

## 🔮 Future Enhancements

The project can be extended by:

* Implementing **Q-Learning**.
* Implementing **Deep Q-Networks (DQN)**.
* Using real-time hotel demand data.
* Adding occupancy and room availability.
* Including competitor pricing.
* Creating a more realistic revenue-based reward function.
* Optimizing the pricing policy automatically.
* Deploying the model using **Streamlit**.
* Creating an interactive hotel pricing dashboard.
* Comparing MDP, Q-Learning and DQN performance.

---

## 💡 Business Impact

A dynamic pricing system can help hotels:

* Increase revenue during high-demand periods.
* Improve occupancy during low-demand periods.
* Reduce revenue loss from inappropriate pricing.
* Make pricing decisions based on historical data.
* Adapt room prices according to changing demand.
* Support automated pricing decisions.

---

## 🏁 Conclusion

This project demonstrates how **Machine Learning and Reinforcement Learning concepts can be combined for hotel demand analysis and dynamic pricing**.

The Machine Learning component identifies booking and cancellation patterns, while the MDP component converts demand information into pricing decisions.

The overall system provides a foundation for building an intelligent **hotel revenue management and dynamic pricing system**.

---

## 👩‍💻 Author

**Sherin John**

Data Science & AI Project

GitHub: `Sherinjohn6`

---

## 📜 Project Type

**Infotact Solutions – Data Science / Machine Learning Project**

**Domain:** Travel & Hospitality
**Application:** Hotel Demand Prediction & Dynamic Pricing
**Techniques:** Machine Learning + MDP + Dynamic Pricing
