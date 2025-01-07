
from sentence_transformers import SentenceTransformer
import supabase

# Load a pre-trained SentenceTransformer model
# Generates 768-dimensional embeddings
model = SentenceTransformer('all-mpnet-base-v2')

# Initialize Supabase client
SUPABASE_URL = "https://djxskdtpvrvprdncsbvt.supabase.co"  # Your Supabase URL
SUPABASE_KEY = ""  # Your Supabase service role key
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)


def get_1536_embedding(sentence):
    """Generate a 1536-dimensional embedding for a single sentence."""
    embedding_768 = model.encode(
        [sentence])[0]  # Get a 768-dimensional embedding
    # Duplicate to reach 1536 dimensions
    embedding_1536 = embedding_768.tolist() + embedding_768.tolist()
    return embedding_1536


def search_nearest_vector(query_embedding, match_threshold=0.50, match_count=1):
    """Query Supabase for the nearest vector match."""
    response = supabase_client.rpc(
        'match_documents',  # Correct function name
        {
            'query_embedding': query_embedding,  # Ensure this is a 1536-dimension vector
            'filter': {},                        # Adjust if you need a specific filter
            'match_count': match_count           # Number of matches to retrieve
        }
    ).execute()
    return response


# Example Usage
query_sentence = "scrimba"
query_embedding = get_1536_embedding(query_sentence)

response = search_nearest_vector(
    query_embedding, match_threshold=0.50, match_count=4)
print(response)
if response.data:
    print("Nearest Matches:")
    for result in response.data:
        print(
            f"ID: {result['id']}, Content: {result['content']}, Similarity: {result['similarity']:.2f}")
else:
    print("No matches found or an error occurred.")
