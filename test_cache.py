#!/usr/bin/env python3
"""
Test script to demonstrate the cache performance improvements
"""
import time
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from call_bridge_trnascription import ResponseCache, load_cache_from_file

def test_cache_performance():
    """Test cache vs AI response times"""
    
    # Load cache
    cache_data = load_cache_from_file()
    cache = ResponseCache(cache_data)
    
    # Test transcripts that should hit cache
    test_transcripts = [
        "We're doing this on our own",  # FSBO Layer 1
        "We had agents call already",   # FSBO Layer 2
        "We're getting some traction already",  # FSBO Layer 3
        "We're still with our agent",   # Expired Layer 1
        "We're not ready to talk",      # Expired Layer 2
        "It's a rental property",       # Absentee Layer 2
    ]
    
    print("🧪 Testing Cache Performance")
    print("=" * 50)
    
    total_cache_time = 0
    cache_hits = 0
    
    for i, transcript in enumerate(test_transcripts, 1):
        print(f"\nTest {i}: '{transcript}'")
        
        # Test cache response time
        start_time = time.time()
        cached_response = cache.get_cached_response(transcript)
        cache_time = time.time() - start_time
        
        if cached_response:
            print(f"✅ Cache Hit: {cached_response}")
            print(f"⚡ Cache Time: {cache_time:.4f} seconds")
            total_cache_time += cache_time
            cache_hits += 1
        else:
            print(f"❌ Cache Miss")
        
        print("-" * 40)
    
    # Summary
    print(f"\n📊 Performance Summary:")
    print(f"Cache Hits: {cache_hits}/{len(test_transcripts)}")
    if cache_hits > 0:
        avg_cache_time = total_cache_time / cache_hits
        print(f"Average Cache Response Time: {avg_cache_time:.4f} seconds")
        print(f"Estimated AI Response Time: ~1-3 seconds")
        print(f"Speed Improvement: ~{(2/avg_cache_time):.0f}x faster")
    
    # Final cache stats
    stats = cache.get_cache_stats()
    print(f"\nCache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_cache_performance()
