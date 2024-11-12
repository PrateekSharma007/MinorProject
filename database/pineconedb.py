import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone  # Updated import for Pinecone
from pinecone import Pinecone as PineconeClient
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


def upload_transcripts_to_pinecone(transcripts_dir, pinecone_api_key, index_name, openai_api_key):
    """
    Uploads transcribed text files to Pinecone database.
    
    Args:
        transcripts_dir (str): Directory containing the transcribed text files
        pinecone_api_key (str): Your Pinecone API key
        index_name (str): Name of the Pinecone index
        openai_api_key (str): Your OpenAI API key
    """
    # Initialize Pinecone client
    pc = PineconeClient(api_key=pinecone_api_key)
    
    # Check if index exists
    if index_name not in pc.list_indexes():
        print(f"Index '{index_name}' not found. Please create it first.")
        return
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Initialize Pinecone vector store
    pinecone_vectorstore = Pinecone(client=pc, index_name=index_name, embedding_function=embeddings.embed_query)
    
    # Text splitter for creating chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    
    # Get all text files in the transcripts directory
    transcript_files = list(Path(transcripts_dir).glob("*.txt"))
    print(f"Found {len(transcript_files)} transcript files")
    
    # Process each transcript file
    for transcript_file in transcript_files:
        print(f"\nProcessing: {transcript_file.name}")
        
        try:
            # Read the transcript
            with open(transcript_file, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Split text into chunks
            texts = text_splitter.split_text(text)
            print(f"Created {len(texts)} text chunks")
            
            # Create metadata for each chunk
            metadatas = [
                {
                    "source": transcript_file.name,
                    "chunk": i,
                    "text": text[:100] + "..."  # First 100 chars as preview
                }
                for i, text in enumerate(texts)
            ]
            
            # Upload to Pinecone
            print("Uploading to Pinecone...")
            pinecone_vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            print(f"Successfully uploaded {transcript_file.name} to Pinecone")
            
        except Exception as e:
            print(f"Error processing {transcript_file.name}: {e}")
    
    print("\nAll transcripts processed!")

def create_pinecone_index(api_key, index_name):
    """
    Creates a new Pinecone index if it doesn't exist.
    """
    pc = PineconeClient(api_key=api_key)
    
    # List existing indexes
    existing_indexes = pc.list_indexes()
    
    # Check if the index exists
    if index_name not in existing_indexes:
        print(f"Creating new index: {index_name}")
        
        # Create a new index with necessary specifications
        pc.create_index(
            name=index_name,
            dimension=1536,  # Embedding dimension for OpenAI embeddings (adjust if necessary)
            metric="cosine"  # Similarity metric
        )
        print(f"Index '{index_name}' created successfully!")
    else:
        print(f"Index '{index_name}' already exists")

if __name__ == "__main__":
    # Your configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Ensure the Pinecone API key is set in the environment
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure the OpenAI API key is set in the environment
    INDEX_NAME = "rag-chatmodel"              # Replace with your index name
    
    # Path to transcriptions directory
    TRANSCRIPTS_DIR = r"C:\Users\sharm\MinorProject\backend\audio_chunks\transcriptions"
    
    try:
        # First, ensure the index exists
        create_pinecone_index(PINECONE_API_KEY, INDEX_NAME)
        
        # Then upload the transcripts
        upload_transcripts_to_pinecone(
            TRANSCRIPTS_DIR,
            PINECONE_API_KEY,
            INDEX_NAME,
            OPENAI_API_KEY
        )
    except Exception as e:
        print(f"An error occurred: {e}")
