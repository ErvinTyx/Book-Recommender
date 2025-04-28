import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def generate_realistic_user_ratings(books_df, n_users=500, output_file="user_ratings.csv"):
    """
    Generate realistic user ratings based on user preferences and book characteristics.
    
    Parameters:
    - books_df: DataFrame containing book data
    - n_users: Number of users to generate
    - output_file: CSV file to save the ratings
    
    Returns:
    - DataFrame with user_id, book_id, and rating columns
    """
    print(f"Generating realistic ratings for {n_users} users...")
    
    # Check if we have required columns
    required_cols = ['isbn13', 'title', 'authors', 'average_rating']
    for col in required_cols:
        if col not in books_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Check if we have category information, and create main_category if needed
    if 'main_category' not in books_df.columns:
        print("'main_category' column not found. Looking for alternative category columns...")
        category_options = ['categories', 'simple_categories', 'category', 'genre']
        category_col = None
        
        for col in category_options:
            if col in books_df.columns:
                category_col = col
                print(f"Using '{col}' for category information.")
                # Extract main category (first category listed)
                books_df['main_category'] = books_df[col].apply(
                    lambda x: str(x).split(',')[0].strip() if pd.notnull(x) and len(str(x)) > 0 else 'Unknown'
                )
                break
        
        if category_col is None:
            print("No category column found. Creating a generic main_category based on available data.")
            # Create a simple category based on other features
            if 'average_rating' in books_df.columns:
                # Create categories based on rating ranges
                def rating_to_category(rating):
                    if pd.isna(rating) or rating == 0:
                        return 'Unknown'
                    elif rating < 2.5:
                        return 'Low Rated'
                    elif rating < 3.5:
                        return 'Medium Rated'
                    else:
                        return 'Highly Rated'
                
                books_df['main_category'] = books_df['average_rating'].apply(rating_to_category)
            else:
                # If all else fails, assign random categories
                categories = ['Fiction', 'Non-Fiction', 'Mystery', 'Romance', 'Science Fiction', 
                              'Fantasy', 'Biography', 'History', 'Self-Help', 'Other']
                books_df['main_category'] = np.random.choice(categories, size=len(books_df))
    
    # Check for emotion columns
    emotion_cols = ['anger_norm', 'disgust_norm', 'fear_norm', 'joy_norm', 
                    'sadness_norm', 'surprise_norm', 'neutral_norm']
    
    # Check if we have emotion data
    has_emotions = all(col in books_df.columns for col in emotion_cols)
    
    if not has_emotions:
        print("Emotion columns not found. Continuing without emotion data.")
        
        # Check for alternative emotion column naming
        alt_emotion_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
        if all(col in books_df.columns for col in alt_emotion_cols):
            print("Found basic emotion columns. Creating normalized versions.")
            # Create normalized versions
            for col in alt_emotion_cols:
                books_df[f'{col}_norm'] = books_df[col]
            
            # Normalize emotion scores if they exist
            emotion_sum = books_df[alt_emotion_cols].sum(axis=1).replace(0, 1)  # Avoid division by zero
            for col in alt_emotion_cols:
                books_df[f'{col}_norm'] = books_df[col].div(emotion_sum)
                
            has_emotions = True
    
    # Create a dictionary to map categories to numerical IDs
    categories = sorted(books_df['main_category'].unique())
    category_to_id = {cat: i for i, cat in enumerate(categories)}
    
    # Generate users with preferences
    users = []
    for user_id in range(1, n_users + 1):
        # User reading volume - some read more than others
        reading_volume = np.random.choice(['light', 'moderate', 'heavy'], 
                                         p=[0.3, 0.5, 0.2])
        
        # Number of books read based on reading volume
        if reading_volume == 'light':
            n_books = np.random.randint(3, 10)
        elif reading_volume == 'moderate':
            n_books = np.random.randint(10, 25)
        else:  # heavy
            n_books = np.random.randint(25, 50)
        
        # User's preferred categories (1-3 categories)
        n_preferred_cats = np.random.randint(1, 4)
        preferred_categories = np.random.choice(categories, size=n_preferred_cats, replace=False)
        
        # User's rating bias (some users rate higher/lower on average)
        rating_bias = np.random.normal(0, 0.5)
        
        # User's rating consistency (some users are more consistent than others)
        rating_consistency = np.random.uniform(0.5, 1.5)
        
        # Create emotional preference profile if we have emotion data
        emotion_preferences = {}
        if has_emotions:
            for emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']:
                # How much this user likes or dislikes books with this emotion
                # Values > 0 mean the user likes books with this emotion
                emotion_preferences[emotion] = np.random.normal(0, 1)
                
        users.append({
            'user_id': user_id,
            'reading_volume': reading_volume,
            'n_books': n_books,
            'preferred_categories': preferred_categories,
            'rating_bias': rating_bias,
            'rating_consistency': rating_consistency,
            'emotion_preferences': emotion_preferences if has_emotions else None
        })
    
    # Generate ratings
    ratings_data = []
    
    for user in tqdm(users, desc="Generating user ratings"):
        user_id = user['user_id']
        
        # Probability of choosing each book for this user
        book_probs = []
        
        for _, book in books_df.iterrows():
            prob = 1.0  # Base probability
            
            # Increase probability for preferred categories
            if book['main_category'] in user['preferred_categories']:
                prob *= 5.0
            
            # Adjust based on book popularity (average rating)
            avg_rating = book['average_rating']
            if not pd.isna(avg_rating) and avg_rating > 0:
                prob *= (avg_rating / 2.5)  # Normalize around 2.5
            
            # Adjust based on emotional preferences if we have emotion data
            if has_emotions and user['emotion_preferences']:
                emotion_score = 0
                for emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']:
                    emotion_col = f"{emotion}_norm"
                    if emotion_col in book and not pd.isna(book[emotion_col]):
                        # How much the book exhibits this emotion * how much the user likes this emotion
                        emotion_score += book[emotion_col] * (1 + user['emotion_preferences'][emotion])
                
                # Normalize and apply emotion score
                if emotion_score > 0:
                    prob *= (1 + emotion_score / 5)
            
            book_probs.append(prob)
        
        # Normalize probabilities
        book_probs = np.array(book_probs)
        if book_probs.sum() > 0:  # Avoid division by zero
            book_probs = book_probs / book_probs.sum()
        else:
            # If all probabilities are zero, use uniform distribution
            book_probs = np.ones(len(book_probs)) / len(book_probs)
        
        # Select books for this user
        n_books = min(user['n_books'], len(books_df))
        selected_indices = np.random.choice(
            range(len(books_df)), 
            size=n_books, 
            replace=False, 
            p=book_probs
        )
        
        for idx in selected_indices:
            book = books_df.iloc[idx]
            
            # Calculate base rating
            # Books in preferred categories tend to get higher ratings
            base_rating = 3.0  # Start at an average rating
            if book['main_category'] in user['preferred_categories']:
                base_rating += 1.0
            
            # Adjust based on book's average rating from other users
            if not pd.isna(book['average_rating']) and book['average_rating'] > 0:
                base_rating = 0.7 * base_rating + 0.3 * book['average_rating']
            
            # Emotional alignment
            emotion_alignment = 0
            if has_emotions and user['emotion_preferences']:
                for emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']:
                    emotion_col = f"{emotion}_norm"
                    if emotion_col in book and not pd.isna(book[emotion_col]):
                        # Positive alignment if user likes this emotion and book has it
                        emotion_alignment += book[emotion_col] * user['emotion_preferences'][emotion]
                
                # Scale and apply emotional alignment
                emotion_alignment = emotion_alignment / 5  # Scale to reasonable range
            
            # Apply user's rating bias and add some randomness
            final_rating = base_rating + user['rating_bias'] + emotion_alignment
            final_rating += np.random.normal(0, user['rating_consistency'])
            
            # Constrain to valid rating range and round to nearest 0.5
            final_rating = max(1, min(5, final_rating))
            final_rating = round(final_rating * 2) / 2  # Round to nearest 0.5
            
            ratings_data.append({
                'user_id': user_id,
                'book_id': book['isbn13'],
                'book_title': book['title'],
                'rating': final_rating
            })
    
    # Create DataFrame and save to CSV
    ratings_df = pd.DataFrame(ratings_data)
    
    # Save to CSV
    ratings_df.to_csv(output_file, index=False)
    print(f"Generated {len(ratings_df)} ratings from {n_users} users")
    print(f"Saved ratings to {output_file}")
    
    # Return analysis of the data
    analyze_ratings(ratings_df)
    
    return ratings_df

def analyze_ratings(ratings_df):
    """Analyze and visualize the generated ratings."""
    print("\nRatings Analysis:")
    print(f"Total ratings: {len(ratings_df)}")
    print(f"Unique users: {ratings_df['user_id'].nunique()}")
    print(f"Unique books: {ratings_df['book_id'].nunique()}")
    print(f"Average rating: {ratings_df['rating'].mean():.2f}")
    
    # Rating distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings_df['rating'], bins=9, kde=True)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('rating_distribution.png')
    print("Rating distribution plot saved as 'rating_distribution.png'")
    
    # Ratings per user distribution
    ratings_per_user = ratings_df.groupby('user_id').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings_per_user, bins=20, kde=True)
    plt.title('Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.savefig('ratings_per_user.png')
    print("Ratings per user plot saved as 'ratings_per_user.png'")
    
    # Average rating per user
    avg_rating_per_user = ratings_df.groupby('user_id')['rating'].mean()
    plt.figure(figsize=(10, 6))
    sns.histplot(avg_rating_per_user, bins=20, kde=True)
    plt.title('Average Rating per User')
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Users')
    plt.savefig('avg_rating_per_user.png')
    print("Average rating per user plot saved as 'avg_rating_per_user.png'")

def implement_in_recommender_system(ratings_df):
    """
    Replace the synthetic data generation in the original recommendation system
    with our more realistic ratings.
    
    This function shows how to modify the prepare_collaborative_filtering_data function.
    """
    # Create the user-book matrix
    user_book_matrix = ratings_df.pivot(
        index='user_id', 
        columns='book_id', 
        values='rating'
    ).fillna(0)
    
    # Convert to sparse matrix for efficiency
    from scipy.sparse import csr_matrix
    sparse_user_book = csr_matrix(user_book_matrix.values)
    
    return ratings_df, user_book_matrix, sparse_user_book

if __name__ == "__main__":
    # Load the book dataset
    print("Loading book dataset...")
    books_df = pd.read_csv('books_with_emotions.csv')
    
    # Generate ratings
    ratings_df = generate_realistic_user_ratings(
        books_df, 
        n_users=500,  # You can adjust this number
        output_file="realistic_user_ratings.csv"
    )
    
    print("\nTo use these ratings in your recommender system:")
    print("1. Replace the synthetic data generation in prepare_collaborative_filtering_data()")
    print("2. Load the ratings from 'realistic_user_ratings.csv'")
    print("3. Create the user-book matrix and sparse matrix as shown in the implement_in_recommender_system() function")