import os
from typing import List, Dict
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the doc processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size= chunk_size,
            chunk_overlap= chunk_overlap,
            length_function= len,
        )

    def extract_txt_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the odf file
        
        Returns:
            Extracted text as a string
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() +"\n"
                
                return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
        
    def extract_text_from_txt(self, txt_path: str) -> str:
        """
        Extract text from a TXT file.
        
        Args:
            txt_path: Path to the txt file
        
        Returns:
            File content as a string
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        
        except Exception as e:
            raise Exception(f"Error reading TXT {txt_path}: {str(e)}")
        
    def process_doc(self, file_path: str) -> List[Dict]:
        """
        Process a document and return chunks with metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """      
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            raw_text = self.extract_txt_from_pdf(file_path)
        elif file_extension == ".txt":
            raw_text = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        #cleaning the text
        cleaned_text = self.clean_text(raw_text)

        #split into chunks
        chunks = self.text_splitter.split_text(cleaned_text)

        #Create doc chunks with metadata
        doc_chunks = []
        filename = os.path.basename(file_path)

        for i, chunk in enumerate(chunks):
            chunk_id = self.generate_chunk_id(filename, i, chunk)

            doc_chunks.append({
                'id': chunk_id,
                'text': chunk,
                'metadata': {
                    'source': filename,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'file_path': file_path,
                    'chunk_size': len(chunk),
                }
            })
        return doc_chunks
    
    def clean_text(self, text: str) -> str:
        """
        Clean the extracted text by removing extra whitespaces and normalize line breaks.
        
        Args:
            text: Raw text to be cleaned
        
        Returns:
            Cleaned text
        """
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line: # non-empty line
                cleaned_lines.append(cleaned_line)

        return'\n'.join(cleaned_lines)
    
    def generate_chunk_id(self, filename: str, chunk_index: int, chunk_text: str) -> str:
        """
        Generate a unique ID for each text chunk.
        
        Args:
            filename: Name of the file
            index: Index of the chunk
            chunk: Text content of the chunk
        
        Returns:
            Unique ID as a string
        """
        # Create a unique hash of the chunk content
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
        return f"{filename}_{chunk_index}_{content_hash}"
    
    def get_doc_stats(self, file_path: str) -> Dict:
        """
        Get statistics about the document.
        
        Args:
            file_path: Path to the document 
            
        Returns:
            Dictionary with document statistics
        """
        chunks = self.process_doc(file_path)

        total_characters = sum(len(chunk['text']) for chunk in chunks)
        total_words = sum(len(chunk['text'].split()) for chunk in chunks)

        return{
            'filename': os.path.basename(file_path),
            'total_chunks': len(chunks),
            'total_characters': total_characters,
            'total_words': total_words,
            'avg_chunk_size': total_characters / len(chunks) if chunks else 0,
        }
    
if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    # Example: Process a document (you'll need to have a test file)
    # chunks = processor.process_doc("sample.pdf")
    # print(f"Generated {len(chunks)} chunks")
    # print(f"First chunk: {chunks[0]['text'][:100]}...")
    
    print("Document processor initialized successfully!")
    print(f"Chunk size: {processor.chunk_size}")
    print(f"Chunk overlap: {processor.chunk_overlap}")



    