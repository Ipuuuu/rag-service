import streamlit as st
import os
import tempfile
from rag_pipeline import RAGPipeline
import time

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Service MVP",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'documents_uploaded' not in st.session_state:
    st.session_state.documents_uploaded = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag_pipeline():
    """Initialize the RAG pipeline with user settings."""
    if st.session_state.rag_pipeline is None:
        with st.spinner("🔄 Initializing RAG Pipeline..."):
            st.session_state.rag_pipeline = RAGPipeline(
                collection_name="rag_documents",
                persist_directory="./chroma_db",
                chunk_size=st.session_state.get('chunk_size', 1000),
                chunk_overlap=st.session_state.get('chunk_overlap', 200),
                openai_api_key=st.session_state.get('openai_api_key', None)
            )
        st.success("✅ RAG Pipeline initialized!")

def main():
    # Title and description
    st.title("🤖 RAG Service MVP")
    st.markdown("**Week 1 Challenge**: A simple Retrieval-Augmented Generation service")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get('openai_api_key', ''),
            help="Enter your OpenAI API key"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key
        
        # Chunk settings
        st.subheader("📝 Document Processing")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap
        
        # Initialize button
        if st.button("🚀 Initialize RAG Pipeline", type="primary"):
            st.session_state.rag_pipeline = None  # Reset pipeline
            initialize_rag_pipeline()
        
        # System stats
        if st.session_state.rag_pipeline:
            st.subheader("📊 System Stats")
            stats = st.session_state.rag_pipeline.get_system_stats()
            
            st.metric("Documents in DB", stats['vector_store']['total_documents'])
            st.metric("Chunk Size", stats['chunk_settings']['chunk_size'])
            st.metric("Chunk Overlap", stats['chunk_settings']['chunk_overlap'])
            
            # OpenAI status
            if stats['openai_configured']:
                st.success("✅ OpenAI Configured")
            else:
                st.warning("⚠️ OpenAI Not Configured")
            
            # Clear database
            if st.button("🗑️ Clear Knowledge Base", type="secondary"):
                result = st.session_state.rag_pipeline.clear_knowledge_base()
                if result['success']:
                    st.success(result['message'])
                    st.session_state.documents_uploaded = []
                else:
                    st.error(result['message'])
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload documents to build your knowledge base"
        )
        
        if uploaded_files and st.button("📤 Process Documents"):
            if not st.session_state.rag_pipeline:
                st.error("❌ Please initialize the RAG pipeline first!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Process document
                        result = st.session_state.rag_pipeline.ingest_document(tmp_file_path)
                        
                        if result['success']:
                            st.success(f"✅ {result['message']}")
                            
                            # Add to uploaded documents list
                            doc_info = {
                                'name': uploaded_file.name,
                                'stats': result['stats'],
                                'chunks_added': result['chunks_added']
                            }
                            st.session_state.documents_uploaded.append(doc_info)
                            
                            # Show document stats
                            with st.expander(f"📋 Stats for {uploaded_file.name}"):
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Total Chunks", result['stats']['total_chunks'])
                                with col_b:
                                    st.metric("Total Words", result['stats']['total_words'])
                                with col_c:
                                    st.metric("Avg Chunk Size", f"{result['stats']['avg_chunk_size']:.0f}")
                        else:
                            st.error(f"❌ {result['message']}")
                    
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ All documents processed!")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
        
        # Show uploaded documents
        if st.session_state.documents_uploaded:
            st.subheader("📚 Uploaded Documents")
            for doc in st.session_state.documents_uploaded:
                with st.expander(f"📄 {doc['name']}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.text(f"Chunks: {doc['chunks_added']}")
                        st.text(f"Words: {doc['stats']['total_words']}")
                    with col_b:
                        st.text(f"Characters: {doc['stats']['total_characters']}")
                        st.text(f"Avg Chunk: {doc['stats']['avg_chunk_size']:.0f}")
    
    with col2:
        st.header("💬 Chat with Your Documents")
        
        # Chat interface
        if not st.session_state.rag_pipeline:
            st.info("👈 Please initialize the RAG pipeline first!")
        elif not st.session_state.rag_pipeline.get_system_stats()['vector_store']['total_documents']:
            st.info("📄 Upload some documents first to start chatting!")
        else:
            # Query input
            query = st.text_input(
                "Ask a question about your documents:",
                placeholder="e.g., What is the main topic of the document?",
                key="query_input"
            )
            
            col_query1, col_query2 = st.columns([3, 1])
            with col_query1:
                n_results = st.slider("Number of chunks to retrieve", 1, 10, 5)
            with col_query2:
                st.write("")  # Spacing
                ask_button = st.button("🔍 Ask", type="primary")
            
            if (ask_button or query) and query:
                if not st.session_state.rag_pipeline.get_system_stats()['openai_configured']:
                    st.warning("⚠️ OpenAI API key not configured. Please add it in the sidebar.")
                else:
                    with st.spinner("🤔 Thinking..."):
                        result = st.session_state.rag_pipeline.query(query, n_results=n_results)
                    
                    if result['success']:
                        # Display answer
                        st.success("✅ Answer found!")
                        st.markdown(f"**Answer:** {result['answer']}")
                        
                        # Show sources
                        st.markdown(f"**Sources:** {', '.join(result['sources'])}")
                        st.caption(f"Based on {result['chunks_used']} relevant chunks")
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'query': query,
                            'answer': result['answer'],
                            'sources': result['sources'],
                            'chunks_used': result['chunks_used']
                        })
                        
                        # Show relevant chunks in expander
                        with st.expander("🔍 View Relevant Chunks"):
                            for i, chunk in enumerate(result['relevant_chunks']):
                                st.markdown(f"**Chunk {i+1}** (Similarity: {chunk['similarity_score']:.3f})")
                                st.markdown(f"*Source: {chunk['metadata']['source']}*")
                                st.text(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
                                st.markdown("---")
                    else:
                        st.error(result['answer'])
            
            # Chat history
            if st.session_state.chat_history:
                st.subheader("💬 Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                    with st.expander(f"Q: {chat['query'][:50]}..."):
                        st.markdown(f"**Q:** {chat['query']}")
                        st.markdown(f"**A:** {chat['answer']}")
                        st.caption(f"Sources: {', '.join(chat['sources'])} | Chunks: {chat['chunks_used']}")
                
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**RAG Service MVP** - Week 1 LinkedIn Challenge | Built with Streamlit, LangChain, ChromaDB & OpenAI")

if __name__ == "__main__":
    main()