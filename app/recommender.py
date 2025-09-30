import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np


from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

class DualRecommender:
    def __init__(self, users_path, posts_path, engagements_path):
       
        self.users_df = pd.read_csv(users_path)
        self.posts_df = pd.read_csv(posts_path)
        self.engagements_df = pd.read_csv(engagements_path)
        
       
        self._prepare_ml_ranker_features()
        self.ml_ranker_model = self._train_ml_ranker()

      
        self.svd_model = self._train_svd_model()
        
        print("DualRecommender initialized and models are trained.")



    # SECTION 1: MACHINE LEARNING RANKER 
   

    def _prepare_ml_ranker_features(self):
        
        tfidf = TfidfVectorizer(min_df=0.01, max_df=0.8)
        user_interests_matrix = tfidf.fit_transform(self.users_df['top_3_interests'].str.replace(',', ''))
        post_tags_matrix = tfidf.transform(self.posts_df['tags'].str.replace(',', ''))
        self.interest_match_scores = cosine_similarity(user_interests_matrix, post_tags_matrix)

       
        self.user_id_to_idx = {user_id: i for i, user_id in enumerate(self.users_df['user_id'])}
        self.post_id_to_idx = {post_id: i for i, post_id in enumerate(self.posts_df['post_id'])}

        
        self.scaler = StandardScaler()
        self.users_df['age_scaled'] = self.scaler.fit_transform(self.users_df[['age']])
        
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        gender_encoded = self.encoder.fit_transform(self.users_df[['gender']]).toarray()
        self.gender_df = pd.DataFrame(gender_encoded, columns=self.encoder.get_feature_names_out(['gender']))
        self.users_df = pd.concat([self.users_df.reset_index(drop=True), self.gender_df], axis=1)

    def _train_ml_ranker(self):
       
        training_df = self.engagements_df.copy()
        
      
        training_df['interest_score'] = training_df.apply(
            lambda row: self.interest_match_scores[self.user_id_to_idx[row['user_id']], self.post_id_to_idx[row['post_id']]],
            axis=1
        )
        training_df = training_df.merge(self.users_df, on='user_id')
        
       
        features = ['interest_score', 'past_engagement_score', 'age_scaled'] + list(self.gender_df.columns)
        target = 'engagement'
        
        X = training_df[features]
        y = training_df[target]
        
       
        model = LogisticRegression(random_state=42, class_weight='balanced')
        model.fit(X, y)
        return model

    def get_ml_ranker_recommendations(self, user_id, top_n=3):
        user_idx = self.user_id_to_idx.get(user_id)
        if user_idx is None:
            return [], pd.DataFrame()

        
        engaged_posts = set(self.engagements_df[self.engagements_df['user_id'] == user_id]['post_id'])
      
      
        candidate_posts = self.posts_df[~self.posts_df['post_id'].isin(engaged_posts)].copy()
        
       
        user_info = self.users_df.iloc[[user_idx]]
        
        
        prediction_df = pd.DataFrame(index=candidate_posts.index)
        prediction_df['interest_score'] = [self.interest_match_scores[user_idx, self.post_id_to_idx[pid]] for pid in candidate_posts['post_id']]
        for col in ['past_engagement_score', 'age_scaled'] + list(self.gender_df.columns):
            prediction_df[col] = user_info[col].values[0]

        
        probabilities = self.ml_ranker_model.predict_proba(prediction_df)[:, 1]
        candidate_posts['recommendation_score'] = probabilities
        
       
        top_recommendations = candidate_posts.sort_values('recommendation_score', ascending=False).head(top_n)
        return top_recommendations['post_id'].tolist(), top_recommendations

    
    # SECTION 2: COLLABORATIVE FILTERING ("Netflix Prize" Approach)
    

    def _train_svd_model(self):
       
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(self.engagements_df[['user_id', 'post_id', 'engagement']], reader)
        
   
        trainset = data.build_full_trainset()
       
       
        model = SVD(n_factors=50, n_epochs=20, random_state=42)
        model.fit(trainset)
        return model

    def get_collaborative_filtering_recommendations(self, user_id, top_n=3):
       
        all_post_ids = self.posts_df['post_id'].unique()
        
        
        engaged_posts = self.engagements_df[self.engagements_df['user_id'] == user_id]['post_id'].unique()
        
        
        predictions = []
        for post_id in all_post_ids:
            if post_id not in engaged_posts:
            
                predicted_rating = self.svd_model.predict(user_id, post_id).est
                predictions.append((post_id, predicted_rating))
        
       
        predictions.sort(key=lambda x: x[1], reverse=True)
        
      
        top_post_ids = [post_id for post_id, rating in predictions[:top_n]]
        
        
        recommended_posts_details = self.posts_df[self.posts_df['post_id'].isin(top_post_ids)]
        return top_post_ids, recommended_posts_details