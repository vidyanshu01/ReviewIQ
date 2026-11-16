# ReviewIQ – AI-Powered Review Intelligence System



ReviewIQ is a data analytics and NLP project that transforms customer reviews into actionable business insights. It analyzes customer sentiment, identifies product-related aspects, evaluates review quality, detects potentially similar reviews, and presents findings through an interactive Streamlit dashboard.



The project uses the Women's Clothing E-Commerce Reviews dataset to explore customer feedback and understand product experiences across different departments.



## Key Features



- **Exploratory Data Analysis:** Analyze ratings, review lengths, recommendations, missing values, and department-level patterns.

- **Sentiment Analysis:** Classify customer reviews into positive, neutral, and negative sentiment using TF-IDF and Logistic Regression.

- **Aspect-Based Analysis:** Identify customer feedback about fit, design, fabric, comfort, and durability.

- **Review Quality Scoring:** Apply a rule-based scoring system to assess review detail, aspect coverage, experience cues, and specificity.

- **Similar Review Detection:** Identify potentially similar reviews using TF-IDF and cosine similarity.

- **Interactive Dashboard:** Explore insights, filter reviews, search customer feedback, and export filtered data.



## Dashboard



Run the application locally to explore six dashboard sections:



1\. Overview

2\. Sentiment Analysis

3\. Aspect Intelligence

4\. Review Quality

5\. Similar Reviews

6\. Review Explorer



## Technology Stack



| Category | Tools |

|---|---|

| Programming | Python |

| Data Analysis | Pandas, NumPy |

| Machine Learning | Scikit-learn, Logistic Regression |

| NLP | TF-IDF, N-grams, Cosine Similarity |

| Visualization | Plotly, Matplotlib, Seaborn |

| Dashboard | Streamlit |

| Model Storage | Joblib |



## Dataset



**Source:** \[Women's Clothing E-Commerce Reviews – Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)



- Original records: 23,486

- Cleaned records: 22,637

- Original features: 11

- Engineered features: 24



The dataset is not included in this repository. Download it from Kaggle and place the CSV file at:



`data/raw/Womens Clothing E-Commerce Reviews.csv`



## Project Results



### Sentiment Analysis



The sentiment model uses TF-IDF features and Logistic Regression.



| Metric | Result |

|---|---:|

| Accuracy | 81.89% |

| Macro F1-score | 66.14% |

| Training samples | 18,109 |

| Testing samples | 4,528 |



**Class-wise F1-score**



| Sentiment | F1-score |

|---|---:|

| Negative | 60.79% |

| Neutral | 45.28% |

| Positive | 92.34% |



The sentiment labels are derived from customer ratings and serve as proxy labels. They are not independently annotated human sentiment labels. The model's performance should therefore be interpreted in that context.



### Review Quality Analysis



The rule-based review quality scoring produced the following distribution:



| Quality Level | Reviews |

|---|---:|

| Low | 14,131 |

| Medium | 8,461 |

| High | 45 |



The score considers review detail, aspect coverage, experience-related cues, and specificity. It is a heuristic measure, not a trained classifier or a fake-review detector.



### Aspect Intelligence



The project identifies five major product aspects:



| Aspect | Mentions |

|---|---:|

| Fit | 14,664 |

| Design | 10,214 |

| Fabric | 8,531 |

| Comfort | 6,616 |

| Durability | 1,091 |



A review may mention multiple aspects, so the counts overlap.



### Similar Review Detection



The system uses TF-IDF and cosine similarity with a similarity threshold of 0.85 to identify candidate pairs.



Similarities indicate textual resemblance, not confirmed duplicate reviews, plagiarism, or fraudulent activity.



## Installation and Setup



### 1. Clone the repository



```bash

git clone https://github.com/YOUR\_GITHUB\_USERNAME/ReviewIQ.git

cd ReviewIQ

```



### 2. Create and activate a virtual environment



**Windows PowerShell**



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



### 3. Install dependencies



```bash

python -m pip install --upgrade pip

pip install -r requirements.txt

```



### 4. Download the dataset



Download the CSV from Kaggle and place it inside:



```text

data/raw/

```



The expected filename is:



```text

Womens Clothing E-Commerce Reviews.csv

```



### 5. Generate project outputs



Run the pipeline from the project root in this order:



```bash

python -m src.data.prepare\_data

python -m src.analysis.eda

python -m src.models.sentiment\_model

python -m src.analysis.quality\_scoring

python -m src.analysis.aspect\_analysis

python -m src.analysis.duplicate\_detection

```



### 6. Launch the dashboard



```bash

python -m streamlit run app.py

```



Open:



`http://localhost:8501`



## Project Structure



```text

ReviewIQ/

│

├── app.py

├── requirements.txt

├── README.md

├── .gitignore

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│   └── sentiment\_model.joblib

│

├── assets/

│   ├── eda/

│   ├── aspects/

│   └── models/

│       └── sentiment\_metrics.json

│

└── src/

&#x20;   ├── data/

&#x20;   │   └── prepare\_data.py

&#x20;   ├── analysis/

&#x20;   │   ├── eda.py

&#x20;   │   ├── quality\_scoring.py

&#x20;   │   ├── aspect\_analysis.py

&#x20;   │   └── duplicate\_detection.py

&#x20;   └── models/

&#x20;       └── sentiment\_model.py

```



## Skills Demonstrated



- Data cleaning and preprocessing

- Exploratory data analysis (EDA)

- Feature engineering

- Natural Language Processing

- Sentiment classification

- Aspect-based review analysis

- Model evaluation

- Text similarity analysis

- Interactive data visualization

- Dashboard development



## Limitations



- Sentiment labels are rating-derived proxy labels.

- Neutral sentiment is harder for the model to classify accurately.

- Review quality scores are heuristic, not objective measures of authenticity.

- Similarity detection identifies candidates and requires human review.

- Findings reflect the supplied dataset and should not automatically be generalized to all e-commerce customers.



## Future Improvements



- Add topic modeling for discovering emerging customer concerns.

- Improve neutral sentiment classification.

- Add department-level and product-level comparisons.

- Introduce time-based review trend analysis.

- Deploy the dashboard for public access.



---



**ReviewIQ – Turning Customer Reviews into Actionable Insights.**

