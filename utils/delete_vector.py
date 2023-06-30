import pinecone

# Initialize Pinecone with your API key and environment
pinecone.init(
    api_key="92ab4fad-6565-484e-a3c7-2353b9099413", environment="us-central1-gcp"
)

# Select the index from the list of indexes
active_indexes = pinecone.list_indexes()
pinecone_index = pinecone.Index(active_indexes[0])

# Delete all vectors in a namespace
delete_response = pinecone_index.delete(delete_all=True, namespace="fdd")

# Check the delete response if needed
print(delete_response)
