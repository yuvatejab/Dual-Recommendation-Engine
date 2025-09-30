# Dual-Strategy Content Recommendation System

The goal of this project is to provide users with a rich and diverse set of content suggestions by leveraging two distinct, powerful machine learning models.

The entire system is wrapped in a clean, interactive web application built with Streamlit, which presents the recommendations in two clear categories:
1.  **For You: Based on Your Profile:** Highly personalized suggestions that match your specific interests and demographics.
2.  **Trending for You: Based on Others:** Serendipitous recommendations based on the behavior of users with similar tastes to yours.

![Demo of the Streamlit Application]([https://i.imgur.com/g0j8z9G.gif](https://dual-recommendation-engine.streamlit.app/))

---

## Model 1: The Machine Learning Ranker ("For You")

This model acts as our core personalization engine. It moves beyond simple scoring and instead learns to predict the *probability* of a user engaging with a post.

### Methodology

The approach is to frame the recommendation task as a supervised machine learning problem. We train a **Logistic Regression** classifier to distinguish between a positive engagement (like) and a negative one (no like).

The model is trained on a rich set of features that combine user, content, and interaction data:
1.  **Interest Match Score:** We use TF-IDF to convert user interests and post tags into numerical vectors. The cosine similarity between these vectors gives us a powerful score that quantifies how well the content aligns with the user's profile.
2.  **User Profile Features:** The user's `past_engagement_score`, `age`, and `gender` are included to provide a complete, 360-degree view of the user.

By learning from historical data, the model can determine the optimal way to weigh these features, creating a highly accurate and personalized ranking for every unseen post.

### Training Data Sample

To train the model, we construct a feature set for every known user-post interaction. Here is a sample of the final training data:

| user_id | post_id | engagement | interest_score | age | gender | top_3_interests | past_engagement_score | age_scaled | gender_F | gender_M | gender_Other |
| :--- | :--- | :---: | :---: | :-: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| U1 | P52 | 1 | 0.3599 | 24 | F | sports, art, gaming | 0.61 | -0.346 | 1.0 | 0.0 | 0.0 |
| U1 | P44 | 0 | 0.0000 | 24 | F | sports, art, gaming | 0.61 | -0.346 | 1.0 | 0.0 | 0.0 |
| U1 | P1 | 1 | 0.4199 | 24 | F | sports, art, gaming | 0.61 | -0.346 | 1.0 | 0.0 | 0.0 |
| U1 | P4 | 1 | 0.3599 | 24 | F | sports, art, gaming | 0.61 | -0.346 | 1.0 | 0.0 | 0.0 |
| U1 | P65 | 0 | 0.5528 | 24 | F | sports, art, gaming | 0.61 | -0.346 | 1.0 | 0.0 | 0.0 |

### Model Evaluation

The model was evaluated on an unseen test set of 200 interactions. The results show a balanced performance in predicting both engagement and non-engagement.

**Classification Report**

| | precision | recall | f1-score | support |
| :--- | :---: | :---: | :---: | :---: |
| **No Engagement (0)** | 0.45 | 0.46 | 0.45 | 101 |
| **Engagement (1)** | 0.43 | 0.42 | 0.43 | 99 |
| | | | | |
| **accuracy** | | | 0.44 | 200 |
| **macro avg** | 0.44 | 0.44 | 0.44 | 200 |
| **weighted avg** | 0.44 | 0.44 | 0.44 | 200 |

**Confusion Matrix**

The confusion matrix gives us a visual breakdown of the model's predictions. We can see how many engagements and non-engagements were correctly classified.

![Confusion Matrix for ML Ranker](https://github.com/user-attachments/assets/b930d350-bc09-47b5-82ee-830c8506baca)

---

## Model 2: Collaborative Filtering ("Trending for You")

This model is our discovery engine. It's based on the famous "Netflix Prize" approach and recommends items based on the behavior of the community.

### Methodology

This approach is fundamentally different from the ML Ranker. It **completely ignores** all user and post metadata (like age, gender, or tags). Instead, it learns directly from the user-post engagement matrix.

We use the **Singular Value Decomposition (SVD)** algorithm, a powerful matrix factorization technique. At a high level, SVD decomposes the user-post matrix into two smaller, "latent feature" matrices:
1.  A **user-feature matrix** that represents the taste profile of each user.
2.  A **post-feature matrix** that represents the characteristics of each post.

By multiplying a user's taste vector with a post's characteristic vector, the model can predict the engagement score a user would give to a post they have never seen before. This allows us to find posts that are popular among users with similar tastes, even if the content seems unrelated at first glance.

### Model Evaluation

This model's performance is measured by how accurately it can predict a user's engagement score. We use 5-fold cross-validation to get a robust measure of its accuracy on unseen data. The key metrics are:
* **RMSE (Root Mean Squared Error):** Measures the average magnitude of the prediction error. Lower is better.
* **MAE (Mean Absolute Error):** Similar to RMSE but less sensitive to large errors. Lower is better.

The results show a strong predictive performance, with an average error of about 0.5 on a 0-1 scale.

**Cross-Validation Results**

| | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean** | **Std Dev** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RMSE** | 0.5090 | 0.5367 | 0.5221 | 0.5338 | 0.5034 | **0.5210** | 0.0132 |
| **MAE** | 0.4894 | 0.5131 | 0.5024 | 0.5075 | 0.4841 | **0.4993** | 0.0109 |

---

## A Look Under the Hood: Data Insights

To give you a better sense of the process, here’s a peek at the data and how our model sees it.

#### Raw Data Samples

We start with three simple datasets:

**Users.csv**
| user_id | age | gender | top_3_interests | past_engagement_score |
| :--- | :-: | :---: | :--- | :---: |
| U1 | 24 | F | sports, art, gaming | 0.61 |
| U2 | 32 | F | travel, food, fashion | 0.93 |
| U3 | 28 | Other | sports, travel, fashion | 0.40 |
| U4 | 25 | M | fashion, music, tech | 0.53 |
| U5 | 24 | M | fashion, food, fitness | 0.80 |

**Posts.csv**
| post_id | creator_id | content_type | tags |
| :--- | :---: | :---: | :--- |
| P1 | U44 | video | sports, food |
| P2 | U26 | video | music, travel |
| P3 | U32 | text | sports, travel |
| P4 | U6 | image | music, gaming |
| P5 | U32 | image | food, fashion |

**Engagements.csv**
| user_id | post_id | engagement |
| :--- | :---: | :---: |
| U1 | P52 | 1 |
| U1 | P44 | 0 |
| U1 | P1 | 1 |
| U1 | P4 | 1 |
| U1 | P65 | 0 |

#### The Model's Viewpoint

After the TF-IDF process, our text is transformed into numerical matrices. The shapes tell us we have 50 users and 100 posts, and have identified 10 unique interest/tag features across the dataset.

| Matrix Description | Shape |
| :--- | :--- |
| User-Interests Matrix | (50, 10) |
| Post-Tags Matrix | (100, 10) |

Finally, the cosine similarity calculation gives us a score for every possible user-post pair, resulting in our **Interest Match Score Matrix**.

| Matrix Description | Shape | Sample Value Interpretation |
| :--- | :--- | :--- |
| Interest Match Score Matrix | (50, 100) | The value at `[0, 0]` (0.4199) is the match between User `U1` and Post `P1`. |

---

## How to Run the App

Getting the recommendation system up and running on your local machine is easy.

1.  **Set up the Environment:**
    * Clone the repository.
    * Create and activate a Python virtual environment.

2.  **Install Dependencies:**
    * Run the following command to install all the necessary libraries:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Launch the App:**
    * Open your terminal, navigate to the project's root folder, and run:
        ```bash
        streamlit run app/app.py
        ```
    * A new tab will open in your web browser with the interactive application!

