import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from tqdm import tqdm
from collections import defaultdict
from functools import lru_cache
import gc

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Part 1: Data Loading and Preprocessing
def load_data(file_path):
    """Load the book dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {df.shape[0]} books and {df.shape[1]} features.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_user_ratings(file_path):
    """Load user ratings from a CSV file."""
    try:
        ratings_df = pd.read_csv(file_path)
        print(f"Loaded {ratings_df.shape[0]} user ratings for {ratings_df['book_id'].nunique()} unique books.")
        return ratings_df
    except Exception as e:
        print(f"Error loading user ratings: {e}")
        return None

def preprocess_data(df):
    """Clean and preprocess the book dataset."""
    # Drop duplicates
    df = df.drop_duplicates(subset=['isbn13'], keep='first')
    
    # Handle missing values
    df['authors'] = df['authors'].fillna('Unknown Author')
    df['categories'] = df['categories'].fillna('Uncategorized')
    df['description'] = df['description'].fillna('')
    
    # Convert to numeric more efficiently
    numeric_cols = ['published_year', 'average_rating', 'num_pages', 'ratings_count']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Create a clean title field
    df['clean_title'] = df['title'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
    
    # Extract main categories (first category listed)
    df['main_category'] = df['simple_categories'].apply(
        lambda x: str(x).split(',')[0].strip() if pd.notnull(x) and len(str(x)) > 0 else 'Unknown'
    )
    
    # Process emotion columns efficiently
    emotion_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
    
    # Convert to numeric in one step
    df[emotion_cols] = df[emotion_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Normalize emotion scores
    emotion_sum = df[emotion_cols].sum(axis=1).replace(0, 1)  # Avoid division by zero
    
    # Compute all normalized values at once
    norm_cols = [f'{col}_norm' for col in emotion_cols]
    df[norm_cols] = df[emotion_cols].div(emotion_sum, axis=0)
    
    return df

# Part 2: Knowledge Graph Construction
def build_knowledge_graph(df):
    """Build a knowledge graph using book metadata."""
    G = nx.Graph()
    
    # Create book nodes dictionary first (more efficient than adding one by one)
    book_nodes = {
        row['isbn13']: {
            'type': 'book',
            'title': row['title'],
            'year': row['published_year'],
            'rating': row['average_rating']
        } for _, row in df.iterrows()
    }
    
    # Add all book nodes at once
    G.add_nodes_from(book_nodes.items())
    print(f"Added {len(book_nodes)} book nodes")
    
    # Prepare edges and nodes for batch addition
    author_nodes = {}
    author_edges = []
    category_nodes = {}
    category_edges = []
    emotion_nodes = {}
    emotion_edges = []
    book_book_edges = []
    
    # Process authors
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing authors"):
        book_id = row['isbn13']
        authors = str(row['authors']).split(',')
        for author in authors:
            author = author.strip()
            if author and author != 'nan':
                author_id = f"author_{author}"
                author_nodes[author_id] = {'type': 'author', 'name': author}
                author_edges.append((book_id, author_id, {'relationship': 'written_by'}))
    
    # Process categories
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing categories"):
        book_id = row['isbn13']
        if pd.notnull(row['categories']):
            categories = str(row['categories']).split(',')
            for category in categories:
                category = category.strip()
                if category and category != 'nan':
                    category_id = f"category_{category}"
                    category_nodes[category_id] = {'type': 'category', 'name': category}
                    category_edges.append((book_id, category_id, {'relationship': 'belongs_to'}))
    
    # Process emotions
    emotion_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
    emotion_threshold = 0.2  # Only add significant emotions
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing emotions"):
        book_id = row['isbn13']
        for emotion in emotion_cols:
            norm_score = row.get(f'{emotion}_norm', 0)
            if norm_score > emotion_threshold:
                emotion_id = f"emotion_{emotion}"
                emotion_nodes[emotion_id] = {'type': 'emotion', 'name': emotion}
                emotion_edges.append((book_id, emotion_id, {'relationship': 'evokes', 'weight': norm_score}))
    
    # Add all nodes and edges in batches
    G.add_nodes_from(author_nodes.items())
    G.add_edges_from(author_edges)
    print(f"Added {len(author_nodes)} author nodes and {len(author_edges)} author edges")
    
    G.add_nodes_from(category_nodes.items())
    G.add_edges_from(category_edges)
    print(f"Added {len(category_nodes)} category nodes and {len(category_edges)} category edges")
    
    G.add_nodes_from(emotion_nodes.items())
    G.add_edges_from(emotion_edges)
    print(f"Added {len(emotion_nodes)} emotion nodes and {len(emotion_edges)} emotion edges")
    
    # Process book-book similarity (by category)
    category_books = defaultdict(list)
    for _, row in df.iterrows():
        main_cat = row['main_category']
        category_books[main_cat].append(row['isbn13'])
    
    # Connect books in the same category
    for category, books in tqdm(category_books.items(), desc="Processing book-book similarities"):
        if len(books) > 1:
            for i in range(len(books)):
                for j in range(i+1, min(i+11, len(books))):  # Only connect to 10 other books max
                    if books[i] != books[j]:
                        book_book_edges.append((books[i], books[j], {'relationship': 'similar_category', 'weight': 0.5}))
    
    # Add book-book edges in batches
    G.add_edges_from(book_book_edges)
    print(f"Added {len(book_book_edges)} book-book similarity edges")
    
    print(f"Knowledge graph built with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

# Part 3: Collaborative Filtering Implementation
def prepare_collaborative_filtering_data(ratings_df, df):
    """Prepare data for collaborative filtering using real user ratings."""
    # Make sure the ratings dataframe has the right format
    if 'user_id' not in ratings_df.columns or 'book_id' not in ratings_df.columns or 'rating' not in ratings_df.columns:
        print("Error: User ratings dataframe must have 'user_id', 'book_id', and 'rating' columns")
        return None, None, None
    
    # Create a sparse matrix representation
    user_book_matrix = ratings_df.pivot(
        index='user_id', 
        columns='book_id', 
        values='rating'
    ).fillna(0)
    
    # Convert to sparse matrix for efficiency
    sparse_user_book = csr_matrix(user_book_matrix.values)
    
    print(f"Created user-book matrix with {user_book_matrix.shape[0]} users and {user_book_matrix.shape[1]} books")
    
    return ratings_df, user_book_matrix, sparse_user_book

def train_collaborative_filtering_model(sparse_matrix):
    """Train a collaborative filtering model using KNN."""
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(sparse_matrix)
    return model

def get_cf_recommendations(model, sparse_matrix, user_idx, user_book_matrix, df, n_recommendations=5):
    """Get collaborative filtering recommendations for a user."""
    # Get book IDs
    book_ids = user_book_matrix.columns.tolist()
    
    # Get user's already rated books (vectorized approach)
    user_ratings = sparse_matrix[user_idx].toarray()[0]
    user_rated_books = set(book_ids[i] for i, rating in enumerate(user_ratings) if rating > 0)
    
    # Find similar users
    distances, indices = model.kneighbors(
        sparse_matrix[user_idx].reshape(1, -1), 
        n_neighbors=min(5, sparse_matrix.shape[0])  # Handle case with few users
    )
    
    # Find recommendations from similar users
    similar_users = indices.flatten()[1:] if indices.shape[1] > 1 else indices.flatten()  # Exclude the user themselves if possible
    
    # Aggregate book ratings from similar users
    book_scores = defaultdict(list)
    
    # More efficient method of aggregating ratings
    for similar_user in similar_users:
        # Check if the similar_user index is within bounds
        if similar_user < sparse_matrix.shape[0]:
            similar_user_ratings = sparse_matrix[similar_user].toarray()[0]
            # Only process non-zero ratings
            for i in np.nonzero(similar_user_ratings)[0]:
                if i < len(book_ids):  # Make sure index is in range
                    book_id = book_ids[i]
                    if book_id not in user_rated_books:
                        book_scores[book_id].append(similar_user_ratings[i])
    
    # Calculate average ratings
    avg_scores = {book_id: sum(ratings)/len(ratings) for book_id, ratings in book_scores.items()}
    
    # Sort books by score and get top N
    top_recommendations = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    # Get book details
    recommended_books = []
    for book_id, score in top_recommendations:
        book_info = df[df['isbn13'] == book_id]
        if not book_info.empty:
            book_info = book_info.iloc[0]
            recommended_books.append({
                'isbn13': book_id,
                'title': book_info['title'],
                'authors': book_info['authors'],
                'score': score
            })
    
    return recommended_books

# Part 4: Hybrid Recommender System
# Cache book attributes for faster access
@lru_cache(maxsize=1024)
def get_node_neighbors_by_type(G, node_id, node_type):
    """Get all neighbors of a given type for a node in the graph."""
    return {neighbor for neighbor in G.neighbors(node_id) 
            if G.nodes[neighbor].get('type') == node_type}

def get_knowledge_graph_recommendations(G, book_id, df, n_recommendations=5):
    """Get recommendations based on the knowledge graph."""
    if book_id not in G:
        return []
    
    # Get all book nodes once
    all_books = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'book']
    
    # Pre-compute attributes for the source book to avoid repetitive computation
    source_authors = get_node_neighbors_by_type(G, book_id, 'author')
    source_categories = get_node_neighbors_by_type(G, book_id, 'category') 
    source_emotions = get_node_neighbors_by_type(G, book_id, 'emotion')
    
    # Calculate scores for all books in the graph
    scores = {}
    
    for other_book in tqdm(all_books, desc="Calculating book scores", disable=None):
        if other_book == book_id:
            continue
            
        # Initialize score
        score = 0
        
        # Check direct connection (most efficient check first)
        if G.has_edge(book_id, other_book):
            edge_data = G.get_edge_data(book_id, other_book)
            score += edge_data.get('weight', 0.5) * 2  # Direct connections are strongest
            
            # If we have a strong direct connection, we can skip further checks
            if score > 1.0:
                scores[other_book] = score
                continue
        
        # Check common authors
        other_book_authors = get_node_neighbors_by_type(G, other_book, 'author')
        common_authors = source_authors.intersection(other_book_authors)
        score += len(common_authors) * 0.5
        
        # Check common categories
        other_book_categories = get_node_neighbors_by_type(G, other_book, 'category')
        common_categories = source_categories.intersection(other_book_categories)
        score += len(common_categories) * 0.3
        
        # Check common emotions
        other_book_emotions = get_node_neighbors_by_type(G, other_book, 'emotion')
        common_emotions = source_emotions.intersection(other_book_emotions)
        score += len(common_emotions) * 0.2
        
        # Store the score if positive
        if score > 0:
            scores[other_book] = score
    
    # Get top n books efficiently
    top_books = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    # Create lookup table for faster access
    book_info_map = {row['isbn13']: row for _, row in df.iterrows()}
    
    # Get book details
    recommended_books = []
    for book_id, score in top_books:
        if book_id in book_info_map:
            book_info = book_info_map[book_id]
            recommended_books.append({
                'isbn13': book_id,
                'title': book_info['title'],
                'authors': book_info['authors'],
                'score': score
            })
    
    return recommended_books

def hybrid_recommendations(user_id, liked_book_id, df, G, cf_model, sparse_user_book, user_book_matrix, n_recommendations=5):
    """Generate hybrid recommendations combining knowledge graph and collaborative filtering."""
    # Map user_id to index in the user_book_matrix
    user_ids = user_book_matrix.index.tolist()
    if user_id not in user_ids:
        print(f"User ID {user_id} not found in the dataset. Using knowledge graph recommendations only.")
        kg_recs = get_knowledge_graph_recommendations(
            G, liked_book_id, df, n_recommendations=n_recommendations
        )
        
        # Format KG recs to match hybrid format
        for rec in kg_recs:
            rec['source'] = 'Knowledge Graph'
        
        return kg_recs
    
    user_idx = user_ids.index(user_id)
    
    # Get recommendations from both systems
    cf_recs = get_cf_recommendations(
        cf_model, sparse_user_book, user_idx, user_book_matrix, df, n_recommendations=n_recommendations
    )
    
    kg_recs = get_knowledge_graph_recommendations(
        G, liked_book_id, df, n_recommendations=n_recommendations
    )
    
    # Merge and rank the recommendations more efficiently
    all_recs = {}
    
    # Add CF recommendations with a 0.6 weight
    for rec in cf_recs:
        book_id = rec['isbn13']
        all_recs[book_id] = {
            'isbn13': book_id,
            'title': rec['title'],
            'authors': rec['authors'],
            'score': rec['score'] * 0.4,
            'source': 'Collaborative Filtering'
        }
    
    # Add KG recommendations with a 0.4 weight
    for rec in kg_recs:
        book_id = rec['isbn13']
        if book_id in all_recs:
            all_recs[book_id]['score'] += rec['score'] * 0.6
            all_recs[book_id]['source'] = 'Hybrid'
        else:
            all_recs[book_id] = {
                'isbn13': book_id,
                'title': rec['title'],
                'authors': rec['authors'],
                'score': rec['score'] * 0.4,
                'source': 'Knowledge Graph'
            }
    
    # Sort by combined score
    sorted_recs = sorted(all_recs.values(), key=lambda x: x['score'], reverse=True)
    
    return sorted_recs[:n_recommendations]

# Part 5: Evaluation
def evaluate_recommendations(test_data, df, G, cf_model, sparse_user_book, user_book_matrix):
    """Evaluate the recommender system using various metrics with improved debugging."""
    # Initialize metrics
    metrics = {
        'precision_sum': 0,
        'recall_sum': 0,
        'ndcg_sum': 0,
        'test_count': 0,
        'successful_recs': 0,
        'empty_recs': 0,
        'missing_users': 0,
        'book_id_mismatches': 0
    }
    
    # Add debug collection
    debug_info = {
        'actual_books': [],
        'rec_lists': [],
        'seed_books': [],
        'match_status': []
    }
    
    # Filter test data to only include ratings >= 4
    filtered_test_data = test_data[test_data['rating'] >= 4]
    
    print(f"Original test data: {test_data.shape[0]} records")
    print(f"Filtered test data (ratings >= 4): {filtered_test_data.shape[0]} records")
    
    # Ensure consistent string format for IDs
    filtered_test_data['user_id'] = filtered_test_data['user_id'].astype(str).str.strip()
    filtered_test_data['book_id'] = filtered_test_data['book_id'].astype(str).str.strip()
    
    # Get valid user and book IDs
    valid_user_ids = set(user_book_matrix.index.astype(str).str.strip())
    valid_book_ids = set(df['isbn13'].astype(str).str.strip())
    
    # Debug information
    print(f"Number of unique users in test data: {filtered_test_data['user_id'].nunique()}")
    print(f"Number of unique books in test data: {filtered_test_data['book_id'].nunique()}")
    print(f"Number of valid users in training data: {len(valid_user_ids)}")
    
    # Process in smaller batches
    batch_size = 10
    num_batches = (len(filtered_test_data) + batch_size - 1) // batch_size
    
    # Create a book ID to book info mapping for efficient lookups
    book_info_map = {str(row['isbn13']).strip(): row for _, row in df.iterrows()}
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluating recommendations"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(filtered_test_data))
        batch = filtered_test_data.iloc[batch_start:batch_end]
        
        for _, row in batch.iterrows():
            user_id = row['user_id']
            actual_book_id = row['book_id']
            
            # Skip if user or book is not in the training data
            if user_id not in valid_user_ids:
                metrics['missing_users'] += 1
                continue
            
            if actual_book_id not in valid_book_ids:
                metrics['book_id_mismatches'] += 1
                continue
            
            # Get user index
            user_idx = list(user_book_matrix.index).index(user_id)
            
            # Get all books the user has rated
            user_ratings = user_book_matrix.loc[user_id].to_dict()
            rated_books = {book_id: rating for book_id, rating in user_ratings.items() if rating > 0}
            
            # If user has no rated books, skip
            if not rated_books:
                metrics['missing_users'] += 1
                continue
                
            # Use the highest-rated book as seed (more likely to give good recommendations)
            seed_book_id = max(rated_books.items(), key=lambda x: x[1])[0]
            
            # Check if seed book exists in graph
            if seed_book_id not in G:
                # Find alternative seed book
                alt_found = False
                for alt_book_id, _ in sorted(rated_books.items(), key=lambda x: x[1], reverse=True):
                    if alt_book_id in G:
                        seed_book_id = alt_book_id
                        alt_found = True
                        break
                
                if not alt_found:
                    # Use popular book as fallback
                    seed_book_id = df.sort_values('ratings_count', ascending=False).iloc[0]['isbn13']
            
            # Get hybrid recommendations with more weight on knowledge graph
            recs = hybrid_recommendations(
                user_id, 
                seed_book_id,
                df, 
                G, 
                cf_model, 
                sparse_user_book, 
                user_book_matrix, 
                n_recommendations=10
            )
            
            if not recs:
                metrics['empty_recs'] += 1
                continue
                
            metrics['successful_recs'] += 1
            
            # Extract recommendation IDs and ensure consistent format
            rec_ids = [str(rec['isbn13']).strip() for rec in recs]
            
            # Enhanced debugging for the first few records
            if metrics['test_count'] < 10:
                debug_info['actual_books'].append(actual_book_id)
                debug_info['rec_lists'].append(rec_ids)
                debug_info['seed_books'].append(seed_book_id)
                debug_info['match_status'].append(actual_book_id in rec_ids)
                
                # Print verbose debugging information
                print(f"\n--- Debug for test case {metrics['test_count']} ---")
                print(f"User: {user_id}")
                
                # Print actual book info
                if actual_book_id in book_info_map:
                    actual_book = book_info_map[actual_book_id]
                    print(f"ACTUAL BOOK: {actual_book['title']} by {actual_book['authors']}")
                else:
                    print(f"ACTUAL BOOK ID: {actual_book_id} (not found in book_info_map)")
                
                # Print seed book info
                if seed_book_id in book_info_map:
                    seed_book = book_info_map[seed_book_id]
                    print(f"SEED BOOK: {seed_book['title']} by {seed_book['authors']}")
                else:
                    print(f"SEED BOOK ID: {seed_book_id} (not found in book_info_map)")
                
                # Print first few recommendations
                print("TOP RECOMMENDATIONS:")
                for i, rec_id in enumerate(rec_ids[:3], 1):
                    if rec_id in book_info_map:
                        rec_book = book_info_map[rec_id]
                        print(f"{i}. {rec_book['title']} by {rec_book['authors']}")
                    else:
                        print(f"{i}. Unknown book with ID {rec_id}")
                
                print(f"Match status: {'Found in recommendations' if actual_book_id in rec_ids else 'NOT found'}")
                print("-----------------------------------")
            
            # Calculate metrics
            precision = 1 if actual_book_id in rec_ids else 0
            metrics['precision_sum'] += precision
            metrics['recall_sum'] += precision
            
            # Calculate NDCG
            relevance = [1 if book_id == actual_book_id else 0 for book_id in rec_ids]
            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
            idcg = 1.0  # 1/log2(1+1) = 1
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics['ndcg_sum'] += ndcg
            
            metrics['test_count'] += 1
    
    # Calculate average metrics
    test_count = metrics['test_count']
    
    # Print diagnostics
    print("\nEvaluation Diagnostics:")
    print(f"Total test cases processed: {test_count}")
    print(f"Successful recommendation attempts: {metrics['successful_recs']}")
    print(f"Empty recommendation lists: {metrics['empty_recs']}")
    print(f"Users not found in training data: {metrics['missing_users']}")
    print(f"Book ID mismatches: {metrics['book_id_mismatches']}")
    
    if debug_info['match_status']:
        match_count = sum(debug_info['match_status'])
        print(f"\nDetailed debug analysis of first {len(debug_info['match_status'])} cases:")
        print(f"Matched: {match_count}, Unmatched: {len(debug_info['match_status']) - match_count}")
    
    return {
        'precision@10': metrics['precision_sum'] / test_count if test_count > 0 else 0,
        'recall@10': metrics['recall_sum'] / test_count if test_count > 0 else 0,
        'ndcg@10': metrics['ndcg_sum'] / test_count if test_count > 0 else 0,
        'test_count': test_count,
        'diagnostics': metrics,
        'debug_info': debug_info
    }

def better_train_test_split(ratings_df, test_size=0.2, random_state=42):
    """
    Split the ratings data while making sure each user has items in both train and test sets.
    This is a more realistic evaluation scenario.
    """
    # Create a dictionary to store train and test data
    train_data = []
    test_data = []
    
    # Group by user
    user_groups = ratings_df.groupby('user_id')
    
    for user_id, user_ratings in user_groups:
        # Only include users with at least 2 ratings
        if len(user_ratings) < 2:
            train_data.append(user_ratings)
            continue
            
        # Get high-rated items (rating >= 4) for test set
        high_rated = user_ratings[user_ratings['rating'] >= 4]
        
        # If no high-rated items, use all items
        if len(high_rated) == 0:
            high_rated = user_ratings
        
        # If only one high-rated item, add it to test
        if len(high_rated) == 1:
            test_data.append(high_rated)
            # Add all other ratings to train
            train_data.append(user_ratings[~user_ratings.index.isin(high_rated.index)])
            continue
            
        # Split high-rated items
        user_train, user_test = train_test_split(
            high_rated, 
            test_size=min(test_size, 0.5),  # Ensure at least half of items stay in training
            random_state=random_state
        )
        
        # Add to main sets
        train_data.append(user_train)
        
        # Add remaining low-rated items to train set
        low_rated = user_ratings[~user_ratings.index.isin(high_rated.index)]
        if len(low_rated) > 0:
            train_data.append(low_rated)
            
        test_data.append(user_test)
    
    # Combine all data
    train_combined = pd.concat(train_data) if train_data else pd.DataFrame()
    test_combined = pd.concat(test_data) if test_data else pd.DataFrame()
    
    # Print statistics
    print(f"Training set: {len(train_combined)} ratings from {train_combined['user_id'].nunique()} users")
    print(f"Test set: {len(test_combined)} ratings from {test_combined['user_id'].nunique()} users")
    
    return train_combined, test_combined

# Part 6: Gradio UI
def create_gradio_interface(df, G, cf_model, sparse_user_book, user_book_matrix, ratings_df):
    """Create a Gradio interface with improved state management and error handling."""
    # Get unique user IDs from the ratings data
    unique_user_ids = sorted(ratings_df['user_id'].unique().tolist())
    
    # Function to get book titles for a specific user
    def get_user_books(user_id):
        if not user_id:
            return []
        
        # Ensure user_id is treated as a string for comparison
        user_id_str = str(user_id)
        
        # Get all books rated by this user
        user_books = ratings_df[ratings_df['user_id'].astype(str) == user_id_str]
        
        if user_books.empty:
            print(f"No books found for user {user_id}")
            return []
        
        book_titles = []
        for _, row in user_books.iterrows():
            book_id = str(row['book_id']).strip()
            
            # Look up the book title
            book_info = df[df['isbn13'].astype(str) == book_id]
            
            if not book_info.empty:
                title = book_info.iloc[0]['title']
                book_titles.append(f"{title} (Rating: {row['rating']})")
            else:
                print(f"Book ID {book_id} not found in books dataset")
        
        # Return the sorted list directly
        return sorted(book_titles)
    
    # Function to show a user's book history
    def show_user_history(user_id):
        if not user_id:
            return "Please select a user ID to see their book history."
        
        user_id_str = str(user_id)
        user_books = ratings_df[ratings_df['user_id'].astype(str) == user_id_str]
        
        if user_books.empty:
            return f"No book history found for user {user_id}."
        
        history = f"### Book History for User {user_id}\n\n"
        
        for _, row in user_books.iterrows():
            book_id = str(row['book_id']).strip()
            book_info = df[df['isbn13'].astype(str) == book_id]
            
            if not book_info.empty:
                title = book_info.iloc[0]['title']
                authors = book_info.iloc[0]['authors']
                rating = row['rating']
                
                history += f"- **{title}** by {authors} - Rating: {rating}/5.0\n"
        
        return history
    
    # Function to recommend books based on user selection with validation
    def recommend_books(user_id, book_title, num_recommendations):
        if not user_id:
            return "Please select a user ID."
        
        if not book_title:
            return "Please select a book."
        
        # Validate that the book title is actually available
        available_books = get_user_books(user_id)
        if book_title not in available_books:
            # Try to handle common issues with book title format
            # Sometimes there might be extra spaces or formatting differences
            closest_match = None
            for available in available_books:
                if book_title.split(" (Rating")[0].strip() == available.split(" (Rating")[0].strip():
                    closest_match = available
                    break
            
            if closest_match:
                book_title = closest_match
                print(f"Found close match: {book_title}")
            else:
                return f"Error: Book '{book_title}' not found for this user. Please try selecting again."
        
        # Extract book ID from the title
        book_name = book_title.split(" (Rating:")[0].strip()
        
        user_id_str = str(user_id)
        user_books = ratings_df[ratings_df['user_id'].astype(str) == user_id_str]
        
        # Find the specific book in the user's history
        selected_book = None
        for _, row in user_books.iterrows():
            book_id = str(row['book_id']).strip()
            book_info = df[df['isbn13'].astype(str) == book_id]
            
            if not book_info.empty and book_info.iloc[0]['title'] == book_name:
                selected_book = book_id
                break
        
        if not selected_book:
            return f"Could not find book '{book_name}' in the dataset. This might be a data inconsistency issue."
        
        try:
            # Get hybrid recommendations
            recommendations = hybrid_recommendations(
                user_id_str,
                selected_book,
                df,
                G,
                cf_model,
                sparse_user_book,
                user_book_matrix,
                n_recommendations=num_recommendations
            )
            
            if not recommendations:
                return "No recommendations found. Try a different book or user."
            
            # Format results
            result = f"### Recommendations based on '{book_name}'\n\n"
            
            for i, rec in enumerate(recommendations, 1):
                result += f"{i}. **{rec['title']}** by {rec['authors']}\n"
                result += f"   - Recommendation score: {rec['score']:.2f}\n"
                result += f"   - Source: {rec['source']}\n\n"
            
            return result
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return f"Error generating recommendations: {str(e)}\n\n```\n{tb}\n```"
    
    # Function to update book dropdown when user changes
    def update_book_dropdown(user_id):
        books = get_user_books(user_id)
        return gr.update(choices=books, value=None if not books else books[0])

    
    # Create the interface
    with gr.Blocks(title="Book Recommender System") as interface:
        gr.Markdown("# Book Recommender System")
        gr.Markdown("## Find your next great read!")
        
        with gr.Row():
            with gr.Column():
                # User ID selection dropdown
                user_id_dropdown = gr.Dropdown(
                    choices=unique_user_ids,
                    label="Select User ID",
                    value=None
                )
                
                # Book selection dropdown - initially empty
                book_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select a book this user has rated",
                    interactive=True
                )
                
                # Number of recommendations slider
                num_recs = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of recommendations")
                
                # Get recommendations button
                submit_button = gr.Button("Get Recommendations")
            
            with gr.Column():
                # User's book history display
                user_history = gr.Markdown(label="User's Book History")
                
                # Recommendations output
                output_text = gr.Markdown(label="Recommended Books")
        
        # Connect user ID dropdown to user history display
        user_id_dropdown.change(
            show_user_history,
            inputs=user_id_dropdown,
            outputs=user_history
        )
        
        # Connect user ID dropdown to update book dropdown
        user_id_dropdown.change(
            update_book_dropdown,
            inputs=user_id_dropdown,
            outputs=book_dropdown
        )
        
        # Connect recommendation button
        submit_button.click(
            recommend_books, 
            inputs=[user_id_dropdown, book_dropdown, num_recs], 
            outputs=output_text
        )
    
    return interface

# Main function to run the entire pipeline
def main(books_data_path, ratings_data_path):
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data(books_data_path)
    if df is None:
        return
    
    df = preprocess_data(df)
    
    # Ensure isbn13 is string type for consistent comparisons
    df['isbn13'] = df['isbn13'].astype(str)
    
    # Load user ratings
    ratings_df = load_user_ratings(ratings_data_path)
    if ratings_df is None:
        return
    
    # Ensure user_id and book_id are string type for consistent comparisons
    ratings_df['user_id'] = ratings_df['user_id'].astype(str)
    ratings_df['book_id'] = ratings_df['book_id'].astype(str)
    
    # Make sure the books in the ratings exist in the books dataset
    book_ids_in_df = set(df['isbn13'].astype(str))
    print(f"Total books in dataset: {len(book_ids_in_df)}")
    
    books_in_ratings = set(ratings_df['book_id'].astype(str))
    print(f"Unique books in ratings: {len(books_in_ratings)}")
    
    # Check for missing books
    missing_books = books_in_ratings - book_ids_in_df
    if missing_books:
        print(f"Warning: {len(missing_books)} books in ratings not found in dataset:")
        print(list(missing_books)[:5])  # Show a few examples
    
    # Filter ratings to only include books that exist in the dataset
    original_count = len(ratings_df)
    ratings_df = ratings_df[ratings_df['book_id'].astype(str).isin(book_ids_in_df)]
    filtered_count = len(ratings_df)
    print(f"Filtered out {original_count - filtered_count} ratings for books not in the dataset")
    print(f"After filtering, {filtered_count} ratings remain for books in the dataset")
    
    # 2. Build knowledge graph
    print("Building knowledge graph...")
    G = build_knowledge_graph(df)
    
    # 3. Prepare data for collaborative filtering using real user ratings
    print("Preparing collaborative filtering data from real user ratings...")
    ratings_df, user_book_matrix, sparse_user_book = prepare_collaborative_filtering_data(ratings_df, df)
    
    # 4. Train collaborative filtering model
    print("Training collaborative filtering model...")
    cf_model = train_collaborative_filtering_model(sparse_user_book)
    
    # 5. Evaluate the model if there's enough data
    if ratings_df.shape[0] > 10:  # Arbitrary threshold for evaluation
        print("Evaluating recommender system...")
        
        # Use our improved train/test split function
        train_data, test_data = better_train_test_split(ratings_df, test_size=0.2, random_state=42)
        
        # Free up memory before evaluation
        gc.collect()
        
        evaluation_results = evaluate_recommendations(
            test_data, 
            df, 
            G, 
            cf_model, 
            sparse_user_book, 
            user_book_matrix
        )
        print("Evaluation results:", evaluation_results)
    else:
        print("Not enough ratings data for evaluation. Skipping evaluation step.")
    
    # 6. Create and launch Gradio interface
    print("Creating Gradio interface...")
    interface = create_gradio_interface(df, G, cf_model, sparse_user_book, user_book_matrix, ratings_df)
    interface.launch(share=False)  # Set share=True if you want a public link

# Example usage
if __name__ == "__main__":
    # Paths to your datasets
    books_data_path = "books_with_emotions.csv"  # Your book dataset with emotions
    ratings_data_path = "realistic_user_ratings.csv"  # Your real user ratings
    main(books_data_path, ratings_data_path)