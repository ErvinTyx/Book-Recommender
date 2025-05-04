import pandas as pd
import numpy as np
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class CollaborativeFiltering:
    def __init__(self):
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.books_df = None
        self.users_df = None
        self.ratings_df = None
        self.book_indices = None
        self.user_indices = None
        self.current_recommendations = []
        self.current_page = 0
        self.page_size = 10

    def load_data(self, books_path, users_path, ratings_path):
        print("Loading books data...")
        self.books_df = pd.read_csv(books_path, dtype={'isbn13': str})
        self.users_df = pd.read_csv(users_path)

        print("Loading ratings data...")
        self.ratings_df = pd.read_csv(ratings_path, dtype={'isbn13': str})
        
        print(f"Columns in ratings file: {self.ratings_df.columns.tolist()}")
        
        if 'ratings' in self.ratings_df.columns and 'rating' not in self.ratings_df.columns:
            self.ratings_df = self.ratings_df.rename(columns={'ratings': 'rating'})
        
        print(f"Ratings before cleaning: {len(self.ratings_df)}")
        self.ratings_df = self.ratings_df.dropna(subset=['user_id', 'rating'])
        print(f"Ratings after cleaning: {len(self.ratings_df)}")
        
        self.ratings_df['user_id'] = self.ratings_df['user_id'].astype(int)
        self.ratings_df['rating'] = pd.to_numeric(self.ratings_df['rating'], errors='coerce')
        self.ratings_df['book_id'] = self.ratings_df['isbn13']
        
        print(f"Data Loaded: {len(self.books_df)} books, {len(self.users_df)} users, {len(self.ratings_df)} ratings.")
        print("Sample ratings data:")
        print(self.ratings_df.head())

    def create_user_item_matrix(self, sample_size=None):
        print("Creating user-item matrix...")
        ratings_df = self.ratings_df
        unique_users = ratings_df['user_id'].unique()
        unique_books = ratings_df['book_id'].unique()
        
        self.user_indices = {user: idx for idx, user in enumerate(unique_users)}
        self.book_indices = {book: idx for idx, book in enumerate(unique_books)}
        
        self.user_item_matrix = np.zeros((len(unique_users), len(unique_books)))
        
        count = 0
        for _, row in ratings_df.iterrows():
            user_id = row['user_id']
            book_id = row['book_id']
            if user_id in self.user_indices and book_id in self.book_indices:
                user_idx = self.user_indices[user_id]
                book_idx = self.book_indices[book_id]
                self.user_item_matrix[user_idx, book_idx] = row['rating']
                count += 1
        
        print(f"Added {count} ratings to user-item matrix")

    def compute_user_similarity(self, batch_size=1000):
        print("Computing user similarity...")
        if self.user_item_matrix is None:
            print("ERROR: User-item matrix is empty")
            return
        
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        print("User similarity matrix computed.")

    def get_similar_users(self, user_id, top_n=5):
        print(f"Getting similar users for user {user_id}...")
        if user_id not in self.user_indices:
            print(f"User {user_id} not found.")
            return []
            
        idx = self.user_indices[user_id]
        similarities = self.similarity_matrix[idx]
        
        similar_indices = np.argsort(similarities)[::-1]
        similar_indices = similar_indices[similar_indices != idx][:top_n]
        
        reverse_indices = {idx: user for user, idx in self.user_indices.items()}
        similar_users = [(reverse_indices[i], similarities[i]) for i in similar_indices]
        
        print(f"Similar users found: {similar_users}")
        return similar_users

    def generate_recommendations(self, user_id, max_recommendations=50, threshold=0.0):
        """
        Generate book recommendations but don't display them yet
        """
        try:
            user_id = int(user_id)
        except ValueError:
            print(f"Invalid user ID format: {user_id}")
            return False

        print(f"Looking up recommendations for user {user_id}")

        if user_id not in self.user_indices:
            print(f"User {user_id} not found in user_indices.")
            return False

        user_idx = self.user_indices[user_id]
        similarities = self.similarity_matrix[user_idx]

        similar_users = np.argsort(similarities)[::-1]
        similar_users = similar_users[similar_users != user_idx][:max_recommendations]

        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        user_books = set(user_ratings['book_id'].tolist())
        print(f"User has rated {len(user_books)} books")

        candidates = {}
        for sim_user_idx in similar_users:
            sim_user_id = list(self.user_indices.keys())[list(self.user_indices.values()).index(sim_user_idx)]
            sim_user_ratings = self.ratings_df[self.ratings_df['user_id'] == sim_user_id]

            for _, rating_row in sim_user_ratings.iterrows():
                book_id = rating_row['book_id']
                rating = float(rating_row['rating'])

                if book_id in user_books:
                    continue  

                if book_id not in candidates:
                    candidates[book_id] = {'total': 0.0, 'sim_sum': 0.0}

                sim_score = float(similarities[sim_user_idx])
                candidates[book_id]['total'] += sim_score * rating
                candidates[book_id]['sim_sum'] += sim_score

        print(f"Found {len(candidates)} candidate books for recommendation")

        recommendations = []
        for book_id, data in candidates.items():
            if data['sim_sum'] > 0:
                predicted_rating = data['total'] / data['sim_sum']
                predicted_rating = min(max(predicted_rating, 1), 5)

                if predicted_rating >= threshold:
                
                    book_info = self.books_df[self.books_df['isbn13'] == book_id]

                    if not book_info.empty:
                        book_title = book_info['title'].iloc[0]
                        authors = book_info['authors'].iloc[0] if 'authors' in book_info.columns else "Unknown"
                        category = book_info['categories'].iloc[0] if 'categories' in book_info.columns else "Unknown"
                        thumbnail = book_info['thumbnail'].iloc[0] if 'thumbnail' in book_info.columns else "cover-not-found.jpg"
                    else:
                        book_title = "Unknown Title"
                        authors = "Unknown"
                        category = "Unknown"
                        thumbnail = "cover-not-found.jpg"

                    recommendations.append({
                        'book_id': book_id,
                        'title': book_title,
                        'authors': authors,
                        'category': category,
                        'thumbnail': thumbnail,
                        'rating': predicted_rating,
                    })

        # Sort recommendations by predicted rating in descending order
        recommendations.sort(key=lambda x: x['rating'], reverse=True)
        recommendations = recommendations[:max_recommendations]

        # Store recommendations for pagination
        self.current_recommendations = recommendations
        self.current_page = 0
        
        return len(recommendations) > 0

    def get_paginated_recommendations(self, page=0):
        """
        Get a specific page of recommendations
        """
        start_idx = page * self.page_size
        end_idx = start_idx + self.page_size
        
        page_recommendations = self.current_recommendations[start_idx:end_idx]
        
        gallery_items = []
        for rec in page_recommendations:
            # Ensure thumbnail is a valid string
            thumbnail = str(rec['thumbnail']) if rec['thumbnail'] else "cover-not-found.jpg"
            # Handle potential missing data with sensible defaults
            title = str(rec['title']) if rec['title'] else "Unknown Title"
            authors = str(rec['authors']) if rec['authors'] else "Unknown Author"
            category = str(rec['category']) if rec['category'] else "Unknown Category"
            rating = round(float(rec['rating']), 2) if rec['rating'] else 0.0
            
            gallery_items.append((
                thumbnail,
                f"{title} by {authors} - {category} (Predicted Rating: {rating})"
            ))

        if not gallery_items:
            return [("cover-not-found.jpg", "No more recommendations found.")]

        return gallery_items
        
    def recommend_books(self, user_id, max_recommendations=50, threshold=0.0):
        """
        Generate recommendations and return the first page
        """
        success = self.generate_recommendations(user_id, max_recommendations, threshold)
        
        if not success:
            return [("cover-not-found.jpg", "No recommendations found or invalid user ID.")]
            
        return self.get_paginated_recommendations(0)
        
    def get_next_page(self):
        """
        Get the next page of recommendations
        """
        if not self.current_recommendations:
            return [("cover-not-found.jpg", "No recommendations available.")]
            
        self.current_page += 1
        max_pages = len(self.current_recommendations) // self.page_size
        
        if self.current_page > max_pages:
            self.current_page = max_pages
            
        return self.get_paginated_recommendations(self.current_page)

cf = CollaborativeFiltering()

cf.load_data("books_cleaned.csv", "users.csv", "user_ratings_realistic.csv")

cf.create_user_item_matrix(sample_size=1000)

cf.compute_user_similarity(batch_size=500)

with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
    gr.Markdown("# 📚 Book Recommender System")

    with gr.Tab("👤 Collaborative Recommender"):
        gr.Markdown("### Recommend books based on similar users")
        user_input = gr.Textbox(label="Enter User ID", placeholder="e.g. 1")
        collab_button = gr.Button("Get Recommendations")
        more_button = gr.Button("Show More Books")
        collab_output = gr.Gallery(label="Recommendations", columns=5, rows=2)
        status_text = gr.Textbox(label="Status", value="Enter a user ID to get recommendations", interactive=False)
        
        def update_status(gallery_items):
            if gallery_items and gallery_items[0][1] != "No recommendations found or invalid user ID.":
                current_page = cf.current_page + 1
                total_pages = (len(cf.current_recommendations) - 1) // cf.page_size + 1
                return f"Showing page {current_page} of {total_pages} ({len(gallery_items)} books of {len(cf.current_recommendations)} total recommendations)"
            elif not gallery_items or gallery_items[0][1] == "No recommendations found or invalid user ID.":
                return "No recommendations found. Try a different user ID."
            else:
                return "No more recommendations available."
        
        collab_button.click(
            fn=lambda uid: cf.recommend_books(uid), 
            inputs=[user_input], 
            outputs=[collab_output]
        ).then(
            fn=update_status,
            inputs=[collab_output],
            outputs=[status_text]
        )
        
        more_button.click(
            fn=lambda: cf.get_next_page(),
            inputs=[],
            outputs=[collab_output]
        ).then(
            fn=update_status,
            inputs=[collab_output],
            outputs=[status_text]
        )
        
if __name__ == "__main__":
    dashboard.launch(share=True)