import os
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def scrape_url_to_rows(url):
    """Scrapes a URL and returns clean contextual text rows from tables or text blocks."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Strip code noise
        for s in soup(["script", "style", "header", "footer", "nav"]): 
            s.extract()
            
        rows_extracted = []
        
        # 1. Look for tables (crucial for government scheme directories)
        for table in soup.find_all('table'):
            for row in table.find_all('tr'):
                cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                if cells:
                    combined_row = " | ".join(cells)
                    if len(combined_row) > 30:
                        rows_extracted.append(combined_row)
                        
        # 2. Fallback to list items or paragraphs if no tables exist
        if not rows_extracted:
            for item in soup.find_all(['li', 'p']):
                text = item.get_text(strip=True)
                if len(text) > 40:
                    rows_extracted.append(text)
                    
        return rows_extracted
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

def main():
    print("🚀 Starting Buildathon Ingestion Pipeline...")
    
    # Expand this list to include all target government scheme URLs
    target_urls = [
        "https://www.tn.gov.in/scheme_list.php?dep_id=Mg==",  # MSME/Agri Snippets
        "https://www.tn.gov.in/scheme_details.php?id=MTU2Ng=="   # Social Welfare Department
    ]
    
    all_documents = []
    
    for url in target_urls:
        print(f"📥 Indexing URL: {url}")
        text_rows = scrape_url_to_rows(url)
        for row in text_rows:
            all_documents.append(Document(page_content=row, metadata={"source": url}))
            
    print(f"📋 Total structured text nodes scraped: {len(all_documents)}")
    
    if not all_documents:
        print("❌ Ingestion aborted: No documents found.")
        return

    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents)
    print(f"✂️ Created {len(chunks)} contextual vector chunks.")

    # Generate and Persist Embeddings locally
    print("🧠 Generating Vector Embeddings via HuggingFace...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("💾 Saving Knowledge Base to './chroma_db' folder...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ Ingestion Phase Complete! Vector storage is permanently saved.")

if __name__ == "__main__":
    main()