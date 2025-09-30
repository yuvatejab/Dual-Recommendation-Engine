import streamlit as st
import pandas as pd
from recommender import DualRecommender
from pathlib import Path


st.set_page_config(
    page_title="Dual-Strategy Recommendation System",
    layout="wide"
)


try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
USERS_PATH = DATA_DIR / "Users.csv"
POSTS_PATH = DATA_DIR / "Posts.csv"
ENGAGEMENTS_PATH = DATA_DIR / "Engagements.csv"


@st.cache_resource
def load_recommender():
  
    recommender = DualRecommender(
        users_path=USERS_PATH,
        posts_path=POSTS_PATH,
        engagements_path=ENGAGEMENTS_PATH
    )
    return recommender


st.title(" Dual-Strategy Content Recommender")
st.markdown("This system provides two types of recommendations: one based on your personal profile and another based on the behavior of users with similar tastes.")


recommender = load_recommender()
users_df = recommender.users_df

st.write("---")

user_list = users_df['user_id'].unique()
selected_user = st.selectbox("First, please select your User ID to get personalized recommendations:", user_list)

if selected_user:
    st.write("---")
    
    st.header(f"👤 User Profile: {selected_user}")
    user_info = users_df[users_df['user_id'] == selected_user].iloc[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Top Interests:** {user_info['top_3_interests']}")
    with col2:
        st.warning(f"**Past Engagement Score:** {user_info['past_engagement_score']:.2f}")

    st.write("---")
    st.header(" Your Personalized Recommendations")

   
    rec_col1, rec_col2 = st.columns(2)

   
    with rec_col1:
        st.subheader("For You: Based on Your Profile")
        st.markdown("_These posts are recommended because they match your interests and demographics._")
        
        ml_ids, ml_details = recommender.get_ml_ranker_recommendations(selected_user)
        
        if not ml_ids:
            st.warning("No profile-based recommendations found.")
        else:
            for _, row in ml_details.iterrows():
                with st.container(border=True):
                    st.markdown(f"**Post ID: {row['post_id']}** (`{row['content_type'].capitalize()}`)")
                    st.markdown(f"**Tags:** `{row['tags']}`")

    
    with rec_col2:
        st.subheader("Trending for You: Based on Others")
        st.markdown("_These posts are popular among users with tastes similar to yours._")
        
        cf_ids, cf_details = recommender.get_collaborative_filtering_recommendations(selected_user)
        
        if not cf_ids:
            st.warning("No similarity-based recommendations found.")
        else:
            for _, row in cf_details.iterrows():
                with st.container(border=True):
                    st.markdown(f"**Post ID: {row['post_id']}** (`{row['content_type'].capitalize()}`)")
                    st.markdown(f"**Tags:** `{row['tags']}`")