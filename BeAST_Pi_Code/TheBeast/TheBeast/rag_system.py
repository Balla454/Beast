#!/usr/bin/env python3
"""
TheBeast AI - RAG System for Dataset-Augmented Chat
Retrieval Augmented Generation system that connects Gemma 2B with organized datasets
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import hashlib
import time
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ sentence-transformers or faiss not installed. Installing...")
    EMBEDDINGS_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.error("❌ PyTorch and transformers required for RAG system")

class DatasetKnowledgeBase:
    """Knowledge base that processes and indexes the organized research datasets"""
    
    def __init__(self, dataset_root: str, preload: bool = True):
        self.dataset_root = Path(dataset_root)
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.index = None
        self.embedding_model = None
        self.preload = preload
        
        # Cache directory for persistent storage
        self.cache_dir = self.dataset_root.parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize embedding model
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Embedding model loaded: all-MiniLM-L6-v2")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                self.embedding_model = None
    
    def load_and_process_datasets(self):
        """Load and process all organized datasets into searchable documents"""
        logger.info("📚 Loading and processing datasets...")
        
        # Try to load from cache first
        if self.preload and self._load_from_cache():
            logger.info("✅ Loaded data from cache")
            return
        
        # Process different data types
        self._process_questionnaire_data()
        self._process_documentation()
        self._process_wesad_metadata()
        self._process_eeg_metadata()
        
        logger.info(f"✅ Processed {len(self.documents)} documents from datasets")
        
        # Create embeddings and search index
        if self.embedding_model and self.documents:
            self._create_embeddings()
            self._build_search_index()
            
            # Save to cache for future use
            if self.preload:
                self._save_to_cache()
    
    def _save_to_cache(self):
        """Save processed data to cache for faster loading"""
        try:
            cache_file = self.cache_dir / "knowledge_base_cache.npz"
            index_file = self.cache_dir / "faiss_index.bin"
            metadata_file = self.cache_dir / "metadata.json"
            
            # Save embeddings and documents
            np.savez_compressed(
                cache_file,
                embeddings=self.embeddings,
                documents=self.documents
            )
            
            # Save FAISS index
            if self.index:
                import faiss
                faiss.write_index(self.index, str(index_file))
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f)
            
            logger.info("✅ Cached knowledge base for faster loading")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cache data: {e}")
    
    def _load_from_cache(self):
        """Load processed data from cache"""
        try:
            cache_file = self.cache_dir / "knowledge_base_cache.npz"
            index_file = self.cache_dir / "faiss_index.bin"
            metadata_file = self.cache_dir / "metadata.json"
            
            if not all([cache_file.exists(), index_file.exists(), metadata_file.exists()]):
                return False
            
            # Load embeddings and documents
            cache_data = np.load(cache_file, allow_pickle=True)
            self.embeddings = cache_data['embeddings']
            self.documents = cache_data['documents'].tolist()
            
            # Load FAISS index
            import faiss
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to load from cache: {e}")
            return False
    
    def _process_questionnaire_data(self):
        """Process CSV questionnaire files"""
        questionnaire_dir = self.dataset_root / "Questionnaire_Data"
        
        if not questionnaire_dir.exists():
            return
        
        for csv_file in questionnaire_dir.glob("*.csv"):
            try:
                # Check if file is empty or very small
                if csv_file.stat().st_size < 10:  # Less than 10 bytes
                    logger.warning(f"⚠️ Skipping empty file: {csv_file.name}")
                    continue
                
                # Try to read the CSV with different encodings and error handling
                df = None
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(csv_file, encoding=encoding, on_bad_lines='skip')
                        break
                    except (pd.errors.EmptyDataError, UnicodeDecodeError):
                        continue
                
                if df is None or df.empty or len(df.columns) == 0:
                    logger.warning(f"⚠️ Skipping file with no data: {csv_file.name}")
                    continue
                
                # Create document from questionnaire data
                content = f"Participant questionnaire data from {csv_file.name}:\n"
                content += f"Columns: {', '.join(df.columns.tolist())}\n"
                content += f"Number of responses: {len(df)}\n"
                
                # Add sample data if available
                if len(df) > 0:
                    content += "Sample responses:\n"
                    for idx, row in df.head(3).iterrows():
                        try:
                            content += f"Response {idx + 1}: {dict(row)}\n"
                        except Exception:
                            content += f"Response {idx + 1}: [Data parsing error]\n"
                
                self.documents.append(content)
                self.metadata.append({
                    'source': str(csv_file),
                    'type': 'questionnaire',
                    'participant': csv_file.stem,
                    'file_size': csv_file.stat().st_size
                })
                
                logger.info(f"✅ Processed questionnaire: {csv_file.name}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to process {csv_file}: {e}")
                # Still add metadata for the file even if we can't process it
                content = f"Questionnaire file: {csv_file.name}\n"
                content += f"Status: Could not parse data - {str(e)}\n"
                content += f"File size: {csv_file.stat().st_size} bytes\n"
                
                self.documents.append(content)
                self.metadata.append({
                    'source': str(csv_file),
                    'type': 'questionnaire_error',
                    'participant': csv_file.stem,
                    'file_size': csv_file.stat().st_size,
                    'error': str(e)
                })
    
    def _process_documentation(self):
        """Process documentation and readme files"""
        doc_dir = self.dataset_root / "Documentation"
        
        if not doc_dir.exists():
            return
        
        for doc_file in doc_dir.glob("*"):
            if doc_file.is_file() and doc_file.suffix in ['.txt', '.md', '.pdf']:
                try:
                    if doc_file.suffix in ['.txt', '.md']:
                        content = doc_file.read_text(encoding='utf-8', errors='ignore')
                    else:
                        # For PDF files, create metadata description
                        content = f"Documentation file: {doc_file.name}\n"
                        content += "Type: Research documentation\n"
                        content += f"Size: {doc_file.stat().st_size} bytes\n"
                    
                    self.documents.append(content[:2000])  # Limit content size
                    self.metadata.append({
                        'source': str(doc_file),
                        'type': 'documentation',
                        'filename': doc_file.name,
                        'file_size': doc_file.stat().st_size
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process {doc_file}: {e}")
    
    def _process_wesad_metadata(self):
        """Process WESAD dataset metadata"""
        wesad_dir = self.dataset_root / "WESAD_Data"
        
        if not wesad_dir.exists():
            return
        
        # Create overview document for WESAD dataset
        pkl_files = list(wesad_dir.glob("*.pkl"))
        csv_files = list(wesad_dir.glob("*.csv"))
        
        content = "WESAD (Wearable Stress and Affect Detection) Dataset Overview:\n"
        content += "This dataset contains multimodal physiological and motion data for stress detection.\n"
        content += f"Number of participants: {len(pkl_files)}\n"
        content += f"Data files: {len(pkl_files)} pickle files, {len(csv_files)} questionnaire files\n"
        content += "Sensors used: Chest device (ECG, EMG, EDA, Temp, Resp) + Empatica E4 wrist device\n"
        content += "Research focus: Stress detection using wearable sensors\n"
        content += "Data format: Preprocessed physiological signals in pickle format\n"
        
        # Add participant information
        participants = [f.stem for f in pkl_files]
        content += f"Participants: {', '.join(participants[:10])}"  # Show first 10
        if len(participants) > 10:
            content += f" and {len(participants) - 10} more"
        
        self.documents.append(content)
        self.metadata.append({
            'source': str(wesad_dir),
            'type': 'dataset_overview',
            'dataset': 'WESAD',
            'participant_count': len(pkl_files)
        })
    
    def _process_eeg_metadata(self):
        """Process EEG dataset metadata"""
        eeg_dir = self.dataset_root / "EEG_Data"
        
        if not eeg_dir.exists():
            return
        
        datasets = list(eeg_dir.glob("*"))
        
        content = "EEG (Electroencephalogram) Datasets Overview:\n"
        content += "Collection of brain activity datasets for neuroscience research.\n"
        content += f"Number of datasets: {len(datasets)}\n"
        content += "Available datasets:\n"
        
        for dataset in datasets:
            if 'eye' in dataset.name.lower():
                content += "- EEG Eye State: Brain activity during eye open/closed states\n"
            elif 'epilepsy' in dataset.name.lower():
                content += "- Epilepsy EEG: Brain signals for seizure detection research\n"
            elif 'evoked' in dataset.name.lower():
                content += "- Visual Evoked Potentials: Brain response to visual stimuli\n"
        
        content += "Applications: Brain-computer interfaces, medical diagnosis, neuroscience research\n"
        
        self.documents.append(content)
        self.metadata.append({
            'source': str(eeg_dir),
            'type': 'dataset_overview',
            'dataset': 'EEG',
            'dataset_count': len(datasets)
        })
    
    def _create_embeddings(self):
        """Create embeddings for all documents using batching for efficiency"""
        if not self.embedding_model:
            logger.error("❌ No embedding model available")
            return
        
        logger.info("🔄 Creating embeddings for documents...")
        try:
            # Use batching for better performance
            self.embeddings = self.embedding_model.encode(
                self.documents, 
                batch_size=32,  # Process in batches for efficiency
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Pre-normalize for cosine similarity
            )
            logger.info(f"✅ Created embeddings with batching: {self.embeddings.shape}")
        except Exception as e:
            logger.error(f"❌ Failed to create embeddings: {e}")
    
    def _build_search_index(self):
        """Build FAISS HNSW search index for fast similarity search"""
        if self.embeddings is None:
            return
        
        try:
            import faiss
            logger.info("🔄 Building FAISS HNSW search index...")
            
            # Create HNSW index for better performance on CPU
            dimension = self.embeddings.shape[1]
            
            # Use HNSW (Hierarchical Navigable Small World) for faster searches
            # M=16 connections per node, efConstruction=200 for build quality
            self.index = faiss.IndexHNSWFlat(dimension, 16)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50  # Search parameter for accuracy/speed tradeoff
            
            # Add embeddings to index (already normalized during creation)
            self.index.add(self.embeddings)
            
            logger.info(f"✅ HNSW search index built with {self.index.ntotal} documents")
        except ImportError:
            logger.error("❌ FAISS not available for search indexing")
        except Exception as e:
            logger.error(f"❌ Failed to build search index: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents based on query using batch processing"""
        if not self.embedding_model or not self.index:
            logger.warning("⚠️ Search not available - missing embeddings or index")
            return []
        
        try:
            # Encode query with batching (even for single query for consistency)
            query_embedding = self.embedding_model.encode(
                [query], 
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Search using HNSW index
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Return results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents) and idx != -1:  # -1 indicates no match found
                    results.append({
                        'rank': i + 1,
                        'score': float(score),
                        'content': self.documents[idx],
                        'metadata': self.metadata[idx]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []

class RAGSystem:
    """Main RAG system that combines retrieval with generation"""
    
    def __init__(self, dataset_root: str, model_path: str, preload: bool = True):
        self.dataset_root = dataset_root
        self.model_path = model_path
        self.preload = preload
        
        # Response cache
        self._response_cache = {}
        self._max_cache_size = 100
        
        # Initialize components
        self.knowledge_base = DatasetKnowledgeBase(dataset_root, preload=preload)
        self.tokenizer = None
        self.model = None
        
        # Initialize the system
        self.initialize()
    
    def _get_cache_key(self, query: str, top_k: int = 2) -> str:
        """Generate cache key for query"""
        cache_string = f"{query.lower().strip()}_{top_k}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        return self._response_cache.get(cache_key)
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response with size limit"""
        if len(self._response_cache) >= self._max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
        
        self._response_cache[cache_key] = response
    
    def initialize(self):
        """Initialize the RAG system with preloading"""
        logger.info("🚀 Initializing RAG System...")
        
        # Load knowledge base (with caching)
        start_time = time.time()
        self.knowledge_base.load_and_process_datasets()
        kb_time = time.time() - start_time
        logger.info(f"✅ Knowledge base loaded in {kb_time:.2f}s")
        
        # Preload language model if enabled
        if self.preload:
            start_time = time.time()
            self.load_language_model()
            model_time = time.time() - start_time
            logger.info(f"✅ Language model preloaded in {model_time:.2f}s")
        
        logger.info("✅ RAG System initialized successfully!")
    
    def load_language_model(self):
        """Load the Gemma 2B language model"""
        if not LLM_AVAILABLE:
            logger.error("❌ Language model libraries not available")
            return
        
        try:
            logger.info("🔄 Loading Gemma 2B model...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info("✅ Language model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load language model: {e}")
    
    def generate_rag_response(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """Generate response using RAG approach with caching and context length optimization"""
        
        # Check cache first
        cache_key = self._get_cache_key(query, top_k=2)
        if use_cache:
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                logger.info("✅ Using cached response")
                return cached_response
        
        # Step 1: Retrieve relevant information (reduced from 3 to 2)
        logger.info(f"🔍 Searching for: {query}")
        retrieved_docs = self.knowledge_base.search(query, top_k=2)  # Reduced context
        
        # Step 2: Create optimized context (shorter)
        context = self._create_optimized_context(retrieved_docs)
        augmented_prompt = self._create_augmented_prompt(query, context)
        
        # Step 3: Generate response
        response = self._generate_response(augmented_prompt)
        
        # Prepare result
        result = {
            'query': query,
            'retrieved_documents': retrieved_docs,
            'context_used': context,
            'response': response,
            'sources': [doc['metadata']['source'] for doc in retrieved_docs],
            'cached': False
        }
        
        # Cache the result
        if use_cache:
            self._cache_response(cache_key, result)
        
        return result
    
    def _create_optimized_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Create optimized, shorter context string from retrieved documents"""
        if not retrieved_docs:
            return "No relevant information found."
        
        context = "Research data:\n\n"
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Truncate content to 300 chars (was 500)
            content = doc['content'][:300]
            if len(doc['content']) > 300:
                content += "..."
            
            context += f"Source {i}: {content}\n"
            context += f"Type: {doc['metadata']['type']}\n\n"
        
        return context
    
    def _create_context_from_retrieval(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Create context string from retrieved documents (legacy method)"""
        return self._create_optimized_context(retrieved_docs)
    
    def _create_augmented_prompt(self, query: str, context: str) -> str:
        """Create the optimized augmented prompt for the language model"""
        # Limit context size for faster processing
        max_context_chars = 800  # Reduced from unlimited
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "..."
        
        prompt = f"""You are TheBeast AI. Respond in English only.

Instructions: Be concise and factual. Use the research data provided.

Research Data:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using the language model with optimized settings"""
        if not self.model or not self.tokenizer:
            return "❌ Language model not available for response generation."
        
        try:
            # Tokenize with reduced max length for faster processing
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1200,  # Reduced from 1500
                truncation=True,
                padding=True,
                return_attention_mask=True
            )
            
            # Generate with optimized parameters for speed
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=80,  # Reduced from 120 for faster generation
                    temperature=0.7,  # Slightly higher for more diverse responses
                    do_sample=True,
                    top_p=0.9,  # Increased for faster sampling
                    top_k=50,   # Increased for faster sampling
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Clean and validate response
            response = self._clean_response(response)
            
            return response if response else "I couldn't generate a response based on the provided context."
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def _clean_response(self, response: str) -> str:
        """Clean and validate the response to ensure quality"""
        if not response:
            return ""
        
        # Remove Arabic or non-ASCII characters first
        import re
        response = re.sub(r'[^\x00-\x7F]+', '', response)
        
        # Remove common unwanted patterns
        unwanted_patterns = [
            "User:", "Human:", "Assistant:", "Answer:", "Response:",
            "Context:", "Research:", "Source:", "Note:", "Edit:",
            "English Answer:", "Answer (in English only):"
        ]
        
        for pattern in unwanted_patterns:
            if pattern in response:
                response = response.split(pattern)[0].strip()
        
        # Clean up extra whitespace and newlines
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Remove empty lines at the start
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            response = ' '.join(lines)
        
        # Ensure proper sentence ending
        if response and not response.endswith(('.', '!', '?')):
            # Find the last complete sentence
            sentences = re.split(r'[.!?]+', response)
            if len(sentences) > 1 and sentences[-1].strip() == '':
                # Remove empty last element and reconstruct
                complete_sentences = [s.strip() for s in sentences[:-1] if s.strip()]
                if complete_sentences:
                    response = '. '.join(complete_sentences) + '.'
            elif sentences and sentences[0].strip():
                # Keep the first complete part if it makes sense
                response = sentences[0].strip()
                if not response.endswith(('.', '!', '?')):
                    response += '.'
        
        return response

def install_dependencies():
    """Install required dependencies for RAG system"""
    logger.info("📦 Installing RAG system dependencies...")
    
    try:
        import subprocess
        import sys
        
        packages = [
            "sentence-transformers",
            "faiss-cpu",
            "pandas",
            "numpy"
        ]
        
        for package in packages:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        logger.info("✅ Dependencies installed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Test the RAG system"""
    
    # Check if dependencies are available
    if not EMBEDDINGS_AVAILABLE:
        logger.info("🔧 Installing missing dependencies...")
        if install_dependencies():
            logger.info("✅ Please restart the script to use the RAG system")
            return
        else:
            logger.error("❌ Failed to install dependencies")
            return
    
    # Initialize RAG system with preloading
    dataset_root = "/Users/collinball/Applications/TheBeast/dataset"
    model_path = "/Users/collinball/Applications/TheBeast/organized/models/gemma3n"
    
    rag_system = RAGSystem(dataset_root, model_path, preload=True)
    
    # Test queries
    test_queries = [
        "What datasets are available for stress detection research?",
        "Tell me about the WESAD dataset",
        "What EEG datasets do we have?",
        "How many participants are in the stress detection study?",
        "What sensors were used in the research?"
    ]
    
    print("\n" + "="*60)
    print("🧠 TheBeast AI - RAG System Demo")
    print("="*60)
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        print("-" * 40)
        
        result = rag_system.generate_rag_response(query)
        
        print(f"🤖 Response: {result['response']}")
        
        if result['sources']:
            print("\n📚 Sources consulted:")
            for source in result['sources'][:3]:
                print(f"  • {Path(source).name}")
        print()

if __name__ == "__main__":
    main()
