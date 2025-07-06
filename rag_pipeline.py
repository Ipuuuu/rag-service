import openai
import os
from typing import List, Dict, Any
from doc_processor import DocumentProcessor
from vector_store import VectorStore

class RAGPipeline:
    def __init__(self,
                    collection_name: str = "rag_documents",
                    persist_directory: str = "./chroma_db",
                    chunk_size: int = 1000,
                    chunk_overlap: int = 200,
                    openai_api_key: str = None):
            
        """
        Initialize the RAG pipeline.

        Args:
            collection_name: Name of the vector store collection
            persist_directory: Directory to persist the vector store
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap size between text chunks
            openai_api_key: OpenAI API key for using GPT models (if None, will look for env variable)
        """
        #Initialize document processor
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        #Initialize vector store
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )

        #Initialize OpenAI client
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")    

        if not openai_api_key:
            print("Warning: OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass it as an argument.")

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process & Ingest a document into the vector store.

        Args:
            file_path: Path to the document file (PDF or TXT)
        
        Returns:
            Dictionary containing processing and ingestion results
        """
        try:
            print(f"Processing document: {file_path}")
            
            # Process document into chunks
            chunks = self.document_processor.process_doc(file_path)
            
            # Add chunks to vector store
            self.vector_store.add_docs(chunks)
            
            # Get document stats
            stats = self.document_processor.get_doc_stats(file_path)
            
            return {
                "success": True,
                "message": f"Successfully ingested {len(chunks)} chunks from {os.path.basename(file_path)}",
                "stats": stats,
                "chunks_added": len(chunks)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error ingesting document: {str(e)}",
                "stats": None,
                "chunks_added": 0
            }
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: User query
            n_results: Number of relevant chunks to retrieve
            
        Returns:
            List of relevant chunks
        """
        search_results = self.vector_store.search_similar(query, n_results=n_results)
        return search_results['results']
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], 
                       max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate an answer using OpenAI's GPT model.
        
        Args:
            query: User query
            context_chunks: Relevant document chunks
            max_tokens: Maximum tokens for the response
            temperature: Response creativity (0.0-1.0)
            
        Returns:
            Generated answer
        """
        if not openai.api_key:
            return "OpenAI API key not configured. Please set your API key."
        
        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"Source: {chunk['metadata']['source']}\n{chunk['text']}" 
            for chunk in context_chunks
        ])
        
        # Create the prompt
        prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve relevant chunks and generate answer.
        
        Args:
            question: User question
            n_results: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(question, n_results)
            
            if not relevant_chunks:
                return {
                    "success": False,
                    "answer": "No relevant documents found. Please upload some documents first.",
                    "sources": [],
                    "chunks_used": 0
                }
            
            # Generate answer
            answer = self.generate_answer(question, relevant_chunks)
            
            # Extract sources
            sources = list(set([chunk['metadata']['source'] for chunk in relevant_chunks]))
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "chunks_used": len(relevant_chunks),
                "relevant_chunks": relevant_chunks  # For debugging/transparency
            }
            
        except Exception as e:
            return {
                "success": False,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "chunks_used": 0
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get overall system statistics.
        
        Returns:
            Dictionary with system stats
        """
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "vector_store": vector_stats,
            "chunk_settings": {
                "chunk_size": self.document_processor.chunk_size,
                "chunk_overlap": self.document_processor.chunk_overlap
            },
            "openai_configured": bool(openai.api_key)
        }
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """
        Clear all documents from the knowledge base.
        
        Returns:
            Dictionary with operation result
        """
        try:
            self.vector_store.clear_collection()
            return {
                "success": True,
                "message": "Knowledge base cleared successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error clearing knowledge base: {str(e)}"
            }

# Example usage and testing
if __name__ == "__main__":
    print("Testing RAG Pipeline")
    print("=" * 50)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(collection_name="test_rag")
    
    # Test with sample document (you would replace this with actual file path)
    # For testing, let's add some sample chunks directly
    sample_chunks = [
        {
            'id': 'test_doc1_chunk1',
            'text': 'Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.',
            'metadata': {'source': 'python_guide.txt', 'chunk_index': 0}
        },
        {
            'id': 'test_doc1_chunk2',
            'text': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.',
            'metadata': {'source': 'ml_basics.txt', 'chunk_index': 0}
        }
    ]
    
    # Add test chunks
    rag.vector_store.add_docs(sample_chunks)
    
    # Test query
    print("\n🔍 Testing Query:")
    result = rag.query("What is Python?")
    
    if result["success"]:
        print(f"✅ Answer: {result['answer']}")
        print(f"📚 Sources: {result['sources']}")
        print(f"📄 Chunks used: {result['chunks_used']}")
    else:
        print(f"❌ Query failed: {result['answer']}")
    
    # Get system stats
    print(f"\n📊 System Stats:")
    stats = rag.get_system_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Clean up
    rag.vector_store.delete_collection()
    print("\n✅ Test completed!")

