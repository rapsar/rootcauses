import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_physics_vocabulary(jsonl_file):
    """Load physics vocabulary from your original physics items JSONL file"""
    items = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            items.append(data)
    return items

def prepare_texts_and_metadata(physics_items):
    """Prepare texts for embedding and metadata for visualization"""
    texts = []
    metadata = []
    
    for item in physics_items:
        # Use the item name as the text to embed
        text = item['item']
        texts.append(text)
        
        # Extract attributes for coloring/grouping
        attrs = item['attributes']
        categories = []
        if attrs.get('is_constant', False):
            categories.append('Constant')
        if attrs.get('is_variable', False):
            categories.append('Variable')
        if attrs.get('is_concept', False):
            categories.append('Concept')
        if attrs.get('is_principle', False):
            categories.append('Principle')
        
        # Primary category for coloring
        if categories:
            primary_category = categories[0]  # Use first category
        else:
            primary_category = 'Other'
        
        metadata.append({
            'item': text,
            'description': item.get('description', ''),
            'primary_category': primary_category,
            'all_categories': ', '.join(categories),
            'is_constant': attrs.get('is_constant', False),
            'is_variable': attrs.get('is_variable', False),
            'is_concept': attrs.get('is_concept', False),
            'is_principle': attrs.get('is_principle', False)
        })
    
    return texts, metadata

def create_embeddings(texts, model):
    """Create embeddings using the given model"""
    logging.info(f"Creating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def create_umap_projection(embeddings, n_neighbors=15, min_dist=0.1, random_state=42):
    """Create UMAP projection of embeddings"""
    logging.info("Creating UMAP projection...")
    
    # Optionally standardize embeddings (can help with UMAP)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Create UMAP projection
    umap_reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        metric='cosine'  # Good for sentence embeddings
    )
    
    projection = umap_reducer.fit_transform(embeddings_scaled)
    return projection

def create_interactive_plot(projection, metadata, title, color_by='primary_category'):
    """Create interactive plotly visualization"""
    df = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'item': [m['item'] for m in metadata],
        'description': [m['description'] for m in metadata],
        'primary_category': [m['primary_category'] for m in metadata],
        'all_categories': [m['all_categories'] for m in metadata]
    })
    
    # Create the plot
    fig = px.scatter(
        df, 
        x='x', 
        y='y',
        color=color_by,
        hover_data=['item', 'description', 'all_categories'],
        title=title,
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        width=800,
        height=600
    )
    
    # Customize hover template
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>" +
                      "Categories: %{customdata[2]}<br>" +
                      "Description: %{customdata[1]}<br>" +
                      "<extra></extra>"
    )
    
    # Update layout
    fig.update_layout(
        title_font_size=16,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_side_by_side_comparison(base_projection, finetuned_projection, metadata):
    """Create side-by-side comparison of base vs fine-tuned embeddings"""
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Base Model (all-MiniLM-L6-v2)', 'Fine-tuned Physics Model'),
        horizontal_spacing=0.05
    )
    
    # Prepare data
    df_base = pd.DataFrame({
        'x': base_projection[:, 0],
        'y': base_projection[:, 1],
        'item': [m['item'] for m in metadata],
        'category': [m['primary_category'] for m in metadata],
        'description': [m['description'] for m in metadata]
    })
    
    df_finetuned = pd.DataFrame({
        'x': finetuned_projection[:, 0],
        'y': finetuned_projection[:, 1],
        'item': [m['item'] for m in metadata],
        'category': [m['primary_category'] for m in metadata],
        'description': [m['description'] for m in metadata]
    })
    
    # Get unique categories and colors
    categories = df_base['category'].unique()
    colors = px.colors.qualitative.Set1[:len(categories)]
    color_map = dict(zip(categories, colors))
    
    # Add base model traces
    for category in categories:
        df_cat = df_base[df_base['category'] == category]
        fig.add_trace(
            go.Scatter(
                x=df_cat['x'],
                y=df_cat['y'],
                mode='markers',
                name=category,
                marker=dict(color=color_map[category], size=6),
                text=df_cat['item'],
                hovertemplate="<b>%{text}</b><br>" +
                              "Category: " + category + "<br>" +
                              "<extra></extra>",
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add fine-tuned model traces
    for category in categories:
        df_cat = df_finetuned[df_finetuned['category'] == category]
        fig.add_trace(
            go.Scatter(
                x=df_cat['x'],
                y=df_cat['y'],
                mode='markers',
                name=category,
                marker=dict(color=color_map[category], size=6),
                text=df_cat['item'],
                hovertemplate="<b>%{text}</b><br>" +
                              "Category: " + category + "<br>" +
                              "<extra></extra>",
                showlegend=False  # Don't duplicate legend
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title="Physics Concepts: Base Model vs Fine-tuned Model Comparison",
        title_font_size=16,
        height=600,
        width=1400,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="UMAP Dimension 1")
    fig.update_yaxes(title_text="UMAP Dimension 2")
    
    return fig

def analyze_clustering_quality(embeddings, metadata, model_name):
    """Analyze clustering quality by category"""
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    
    # Get category labels
    categories = [m['primary_category'] for m in metadata]
    unique_categories = list(set(categories))
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    category_labels = [category_to_idx[cat] for cat in categories]
    
    # Calculate silhouette score
    silhouette = silhouette_score(embeddings, category_labels)
    
    print(f"\n{model_name} Clustering Analysis:")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Categories: {len(unique_categories)}")
    for cat in unique_categories:
        count = categories.count(cat)
        print(f"  {cat}: {count} items")
    
    return silhouette

def main():
    # Configuration
    PHYSICS_VOCAB_FILE = "./data/physics_vocabulary_with_attributes_and_description.jsonl"  # Your original physics items file
    BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    # FINETUNED_MODEL_PATH = "physics-embedding-model-20250816_203537"
    FINETUNED_MODEL_PATH = "physics-embedding-model-large-20250817_195614"
    
    # Load physics vocabulary
    logging.info(f"Loading physics vocabulary from {PHYSICS_VOCAB_FILE}")
    physics_items = load_physics_vocabulary(PHYSICS_VOCAB_FILE)
    logging.info(f"Loaded {len(physics_items)} physics items")
    
    # Prepare texts and metadata
    texts, metadata = prepare_texts_and_metadata(physics_items)
    
    # Load models
    logging.info("Loading base model...")
    base_model = SentenceTransformer(BASE_MODEL_NAME)
    
    logging.info("Loading fine-tuned model...")
    finetuned_model = SentenceTransformer(FINETUNED_MODEL_PATH)
    
    # Create embeddings
    base_embeddings = create_embeddings(texts, base_model)
    finetuned_embeddings = create_embeddings(texts, finetuned_model)
    
    # Create UMAP projections
    base_projection = create_umap_projection(base_embeddings, random_state=42)
    finetuned_projection = create_umap_projection(finetuned_embeddings, random_state=42)
    
    # Analyze clustering quality
    analyze_clustering_quality(base_embeddings, metadata, "Base Model")
    analyze_clustering_quality(finetuned_embeddings, metadata, "Fine-tuned Model")
    
    # Create individual plots
    base_fig = create_interactive_plot(
        base_projection, 
        metadata, 
        "Physics Concepts - Base Model (all-MiniLM-L6-v2)"
    )
    
    finetuned_fig = create_interactive_plot(
        finetuned_projection, 
        metadata, 
        "Physics Concepts - Fine-tuned Model"
    )
    
    # Create side-by-side comparison
    comparison_fig = create_side_by_side_comparison(
        base_projection, 
        finetuned_projection, 
        metadata
    )
    
    # Save plots
    base_fig.write_html("physics_base_model_umap.html")
    finetuned_fig.write_html("physics_finetuned_model_umap.html")
    comparison_fig.write_html("physics_models_comparison.html")
    
    logging.info("Plots saved:")
    logging.info("- physics_base_model_umap.html")
    logging.info("- physics_finetuned_model_umap.html") 
    logging.info("- physics_models_comparison.html")
    
    # Show plots
    base_fig.show()
    finetuned_fig.show()
    comparison_fig.show()

if __name__ == "__main__":
    main()

# Additional utility functions

def find_nearest_neighbors(item_name, embeddings, metadata, model_name, top_k=5):
    """Find nearest neighbors for a specific physics concept"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Find the item
    item_idx = None
    for i, meta in enumerate(metadata):
        if meta['item'].lower() == item_name.lower():
            item_idx = i
            break
    
    if item_idx is None:
        print(f"Item '{item_name}' not found")
        return
    
    # Calculate similarities
    target_embedding = embeddings[item_idx:item_idx+1]
    similarities = cosine_similarity(target_embedding, embeddings)[0]
    
    # Get top-k most similar
    top_indices = np.argsort(similarities)[::-1][:top_k+1]  # +1 to exclude self
    
    print(f"\n{model_name} - Nearest neighbors for '{item_name}':")
    for i, idx in enumerate(top_indices):
        if idx == item_idx:
            continue  # Skip self
        item = metadata[idx]['item']
        similarity = similarities[idx]
        category = metadata[idx]['primary_category']
        print(f"{i+1}. {item} (similarity: {similarity:.4f}, category: {category})")

def compare_neighborhoods(item_name, base_embeddings, finetuned_embeddings, metadata):
    """Compare nearest neighbors between base and fine-tuned models"""
    print(f"\nComparing neighborhoods for '{item_name}':")
    print("=" * 50)
    find_nearest_neighbors(item_name, base_embeddings, metadata, "Base Model")
    find_nearest_neighbors(item_name, finetuned_embeddings, metadata, "Fine-tuned Model")

# Example usage after running main():
# compare_neighborhoods("Temperature", base_embeddings, finetuned_embeddings, metadata)
# compare_neighborhoods("Quantum mechanics", base_embeddings, finetuned_embeddings, metadata)
# compare_neighborhoods("Newton's second law", base_embeddings, finetuned_embeddings, metadata)