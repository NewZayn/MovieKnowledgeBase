"""
Testes unitários para classificadores de qualidade
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.quality_classifier import QualityClassifier

@pytest.fixture
def classifier():
    """Fixture para criar classifier"""
    base_dir = Path(__file__).parent.parent
    embeddings_dir = base_dir / 'data/processed/embeddings'
    
    if not embeddings_dir.exists():
        pytest.skip("Embeddings não encontrados. Execute o pipeline primeiro.")
    
    return QualityClassifier(str(embeddings_dir))

def test_classifier_initialization(classifier):
    """Testa inicialização"""
    assert classifier.embeddings is not None
    assert classifier.documents is not None

def test_genre_extraction(classifier):
    """Testa extração de labels de gênero"""
    labels, names = classifier.extract_genre_labels()
    
    assert len(labels) == len(classifier.documents)
    assert len(names) > 0

def test_quality_extraction(classifier):
    """Testa extração de labels de qualidade"""
    labels, names = classifier.extract_rating_labels(bins=3)
    
    assert len(labels) == len(classifier.documents)
    assert len(names) == 3

def test_train_classifier(classifier):
    """Testa treinamento"""
    classifier.extract_genre_labels()
    results = classifier.train_classifier(
        model_type='logistic',
        test_size=0.2
    )
    
    assert 'test_accuracy' in results
    assert 'train_accuracy' in results
    assert 0 <= results['test_accuracy'] <= 1

def test_cross_validation(classifier):
    """Testa validação cruzada"""
    classifier.extract_genre_labels()
    scores = classifier.cross_validate(cv=3, model_type='logistic')
    
    assert len(scores) == 3
    assert all(0 <= s <= 1 for s in scores)
