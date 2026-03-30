#!/usr/bin/env python3
"""
Quick test of the hybrid matching system.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def test_hybrid_matching():
    """Test the hybrid matching with a sample protocol."""
    from src.database import DatabaseManager
    from src.analysis.vector_store import get_vector_store
    from src.analysis.hybrid_matcher import HybridTrialMatcher

    print("=" * 60)
    print("Testing Hybrid Matching System")
    print("=" * 60)

    # Check vector store status
    vector_store = get_vector_store()
    if vector_store.is_initialized():
        count = vector_store.collection.count()
        print(f"\n✓ Vector store has {count:,} trials indexed")
    else:
        print("\n⚠ Vector store is empty - will use keyword search only")

    # Initialize components
    db = DatabaseManager.get_instance()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("✗ ANTHROPIC_API_KEY not found")
        return

    print(f"✓ Database connected")
    print(f"✓ API key configured")

    # Test protocol
    test_protocol = """
    Study Title: A Phase 3 Randomized Study of Drug X vs Placebo in Type 2 Diabetes

    Condition: Type 2 Diabetes Mellitus

    Intervention: Drug X (GLP-1 receptor agonist), 1mg subcutaneous injection weekly

    Primary Endpoint: Change in HbA1c from baseline to Week 26

    Key Inclusion Criteria:
    - Adults 18-75 years old
    - Diagnosis of Type 2 Diabetes for at least 6 months
    - HbA1c 7.0% to 10.0%
    - BMI 25-40 kg/m²

    Key Exclusion Criteria:
    - Type 1 Diabetes
    - eGFR < 30 mL/min/1.73m²
    - History of pancreatitis
    - Pregnant or breastfeeding women
    """

    print("\n" + "-" * 60)
    print("Testing with sample Type 2 Diabetes protocol...")
    print("-" * 60)

    # Initialize hybrid matcher
    matcher = HybridTrialMatcher(db, api_key)

    # Run matching
    use_vector = vector_store.is_initialized() and vector_store.collection.count() > 1000

    print(f"\nUsing vector search: {use_vector}")
    print(f"Using Claude re-ranking: True")
    print("\nSearching for similar trials...")

    query, matches = matcher.find_similar_trials(
        protocol_text=test_protocol,
        min_similarity=30,
        max_results=10,
        use_vector_search=use_vector,
        use_claude_reranking=True
    )

    print(f"\n✓ Found {len(matches)} similar trials")

    # Display results
    print("\n" + "=" * 60)
    print("TOP MATCHING TRIALS:")
    print("=" * 60)

    for i, match in enumerate(matches[:5], 1):
        print(f"\n{i}. {match.nct_id} - Score: {match.overall_similarity:.1f}%")
        print(f"   Title: {match.title[:80]}..." if len(match.title) > 80 else f"   Title: {match.title}")
        print(f"   Vector: {match.vector_score:.2f} | Keyword: {match.keyword_score:.1f} | Eligibility: {match.eligibility_score:.1f}")
        if match.key_similarities:
            print(f"   Key similarities: {', '.join(match.key_similarities[:2])}")

    print("\n" + "=" * 60)
    print("Hybrid Matching Test Complete!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_hybrid_matching()
