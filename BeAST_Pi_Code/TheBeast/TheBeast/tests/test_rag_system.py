import pytest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Mock external heavy dependencies before importing rag_system
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

from rag_system import DatasetKnowledgeBase

@pytest.fixture
def mock_dataset_root(tmp_path):
    """Create a temporary directory structure for datasets"""
    root = tmp_path / "datasets"
    root.mkdir()
    # Create cache dir which is expected to be at parent/cache
    (root.parent / "cache").mkdir()
    return root

def test_knowledge_base_initialization(mock_dataset_root):
    """Test that DatasetKnowledgeBase initializes correctly"""
    kb = DatasetKnowledgeBase(str(mock_dataset_root), preload=False)
    
    assert kb.dataset_root == mock_dataset_root
    assert kb.documents == []
    assert kb.metadata == []
    assert kb.cache_dir == mock_dataset_root.parent / "cache"

def test_cache_directory_creation(tmp_path):
    """Test that cache directory is created if it doesn't exist"""
    root = tmp_path / "datasets"
    root.mkdir()
    # Ensure cache dir doesn't exist
    cache_dir = root.parent / "cache"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        
    kb = DatasetKnowledgeBase(str(root), preload=False)
    assert cache_dir.exists()

@patch('rag_system.logger')
def test_load_and_process_datasets_empty(mock_logger, mock_dataset_root):
    """Test loading datasets when directory is empty"""
    kb = DatasetKnowledgeBase(str(mock_dataset_root), preload=False)
    
    # Mock the internal processing methods to avoid needing real files
    kb._process_questionnaire_data = MagicMock()
    kb._process_documentation = MagicMock()
    kb._process_wesad_metadata = MagicMock()
    kb._process_eeg_metadata = MagicMock()
    kb._create_embeddings = MagicMock()
    kb._build_search_index = MagicMock()
    
    kb.load_and_process_datasets()
    
    # Verify methods were called
    kb._process_questionnaire_data.assert_called_once()
    kb._process_documentation.assert_called_once()
