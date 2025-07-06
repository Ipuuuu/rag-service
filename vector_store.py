import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os
from sentence_transformers import SentenceTransformer
import numpy as np 

class VectorStore:
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store using Chroma.

        Args:
            collection_name: Name of the collection to store documents
            persist_directory: Directory to save the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # create dir if it does not exist
        os.makedirs(self.persist_directory, exist_ok=True)

        #initialize chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)

        #get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        #initializing embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded!")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to generate embeddings for
            
        Returns:
            List of embedding vectors
        """

        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()
    
    def add_docs(self, document_chunks: List[Dict]) -> None:
        """
        Add document chunks to the vector store.

        Args:
            document_chunks: List of documents chunks from DocumentProcessor
        """

        if not document_chunks:
            print("No document chunks to add.")
            return
       
        print(f"Adding {len(document_chunks)} chunks to vector store...")

        #extract texts and metadata
        texts = [chunk['text'] for chunk in document_chunks]
        ids = [chunk['id'] for chunk in document_chunks]
        metadatas = [chunk['metadata'] for chunk in document_chunks]

        #generating embeddings
        embeddings = self.generate_embeddings(texts)

        #add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Successfully added {len(document_chunks)} chunks to vector store!")
    
    def search_similar(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents based on a query.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results
        """

        print(f"Searching for: '{query}'")

        #generate embedding for the query
        query_embedding = self.generate_embeddings([query])[0]

        #search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        #format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': 1 - results['distances'][0][i] #convert distance to similarity
            })
        
        return {
            'query': query,
            'results': formatted_results,
            'total_results': len(formatted_results)
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """

        count = self.collection.count()

        return{
            'collection_name': self.collection_name,
            'total_documents': count,
            'persist_directory': self.persist_directory
        }
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        """
        #get all ids and delete them
        all_items = self.collection.get()
        if all_items['ids']:
            self.collection.delete(ids=all_items['ids'])
            print(f"Cleared {len(all_items['ids'])} documents from collection")
        else:
            print("Collection is already empty")

    def delete_collection(self) -> None:
        """
        Delete the entire collection
        """
        self.client.delete_collection(name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")

#example & testing
if __name__ == "__main__":
    #test vector store 
    print("Testing Vector Store")
    print("=" * 50)

    #initialize vector store
    vector_store = VectorStore(collection_name="test_collection")
    
    #create some sample document chunks (like what DocumentProcessor creates)
    sample_chunks = [
         {
            'id': 'doc1_chunk1',
            'text': 'Artificial Intelligence is transforming how we work and live. Machine learning algorithms can process vast amounts of data.',
            'metadata': {'source': 'ai_guide.txt', 'chunk_index': 0}
        },
        {
            'id': 'doc1_chunk2', 
            'text': 'Natural Language Processing enables computers to understand human language. This technology powers chatbots and virtual assistants.',
            'metadata': {'source': 'ai_guide.txt', 'chunk_index': 1}
        },
        {
            'id': 'doc1_chunk3',
            'text': 'Vector databases store high-dimensional vectors efficiently. They enable fast similarity search for AI applications.',
            'metadata': {'source': 'ai_guide.txt', 'chunk_index': 2}
        }
    ]

    # Add documents to vector store
    vector_store.add_docs(sample_chunks)
    
    # Get collection stats
    stats = vector_store.get_collection_stats()
    print(f"\nCollection Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test search
    print(f"\nTesting Search:")
    search_results = vector_store.search_similar("What is machine learning?", n_results=2)
    
    print(f"Query: {search_results['query']}")
    print(f"Found {search_results['total_results']} results:\n")
    
    for i, result in enumerate(search_results['results']):
        print(f"Result {i+1}:")
        print(f"   Similarity: {result['similarity_score']:.3f}")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Source: {result['metadata']['source']}")
        print()
    
    # Clean up test collection
    vector_store.delete_collection()
    print("✅ Test completed successfully!")

