from doc_processor import DocumentProcessor
import os


def test_document_processor():
    """Test the document processor with a sample text file."""
    
    # Create a sample text file for testing
    sample_text = """
    This is a sample document for testing the RAG service.
    
    Artificial Intelligence (AI) is transforming how we work and live. 
    Machine learning algorithms can now process vast amounts of data 
    and identify patterns that humans might miss.
    
    Natural Language Processing (NLP) is a subset of AI that focuses 
    on the interaction between computers and humans through natural language. 
    It enables computers to understand, interpret, and generate human language.
    
    Retrieval-Augmented Generation (RAG) combines the power of retrieval 
    systems with generative AI models. This approach allows AI systems 
    to access relevant information from a knowledge base and generate 
    more accurate and contextual responses.
    
    Vector databases are essential for RAG systems as they store 
    document embeddings and enable efficient similarity search. 
    Popular vector databases include Chroma, Pinecone, and Weaviate.
    
    The future of AI looks promising with continued advancements 
    in model architectures, training techniques, and computational power.
    """
    
    # Write sample text to a file
    with open("sample_document.txt", "w") as f:
        f.write(sample_text)
    
    # Initialize processor
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
    
    print("Testing Document Processor")
    print("=" * 50)
    
    try:
        # Process the document
        chunks = processor.process_doc("sample_document.txt")
        
        print(f"Successfully processed document!")
        print(f"Generated {len(chunks)} chunks")
        print()
        
        # Display chunks
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}:")
            print(f"   ID: {chunk['id']}")
            print(f"   Length: {len(chunk['text'])} characters")
            print(f"   Text: {chunk['text'][:100]}...")
            print(f"   Metadata: {chunk['metadata']}")
            print("-" * 40)
        
        # Get document statistics
        stats = processor.get_doc_stats("sample_document.txt")
        print("\nDocument Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Clean up
        os.remove("sample_document.txt")
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # Clean up in case of error
        if os.path.exists("sample_document.txt"):
            os.remove("sample_document.txt")

if __name__ == "__main__":
    test_document_processor()