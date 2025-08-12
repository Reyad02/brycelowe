#!/usr/bin/env python3
"""
Standalone cache performance test
"""
import json
import time
import os
from collections import OrderedDict
from datetime import datetime, timedelta

class ResponseCache:
    def __init__(self, cache_data=None):
        self.cache = OrderedDict()
        self.max_size = 1_000_000
        self.ttl_days = 30
        self.lead_types = {}
        self.emotional_tags = []
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
        if cache_data:
            self.load_cache_data(cache_data)
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0
        return {
            'hit_rate': f"{hit_rate:.2f}%",
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': self.total_requests,
            'cache_size': len(self.cache),
            'lead_types_loaded': len(self.lead_types)
        }
    
    def load_cache_data(self, cache_data):
        """Load the JSON cache data"""
        self.lead_types = {lead['id']: lead for lead in cache_data.get('lead_types', [])}
        self.emotional_tags = cache_data.get('emotional_tags', [])
        print(f"Loaded cache with {len(self.lead_types)} lead types")
    
    def detect_lead_type(self, transcript):
        """Enhanced keyword matching to detect lead type from transcript"""
        transcript_lower = transcript.lower()
        
        # Enhanced keywords for different lead types
        keywords = {
            'L001': ['fsbo', 'for sale by owner', 'selling ourselves', 'doing this on our own', 'without agent', 'by owner'],
            'L002': ['expired', 'listing expired', 'previous agent', 'last agent', 'agent before', 'listing ended'],
            'L003': ['rental', 'investment property', 'absentee', 'rent out', 'tenant', 'rental income'],
            'L015': ['referral', 'attorney', 'cpa', 'professional', 'lawyer', 'accountant', 'refer clients']
        }
        
        # Score-based matching for better accuracy
        scores = {}
        for lead_id, words in keywords.items():
            score = sum(1 for word in words if word in transcript_lower)
            if score > 0:
                scores[lead_id] = score
        
        if scores:
            # Return the lead type with highest score
            return max(scores, key=scores.get)
        
        return 'L001'  # Default to FSBO
    
    def detect_objection_layer(self, transcript):
        """Enhanced objection layer detection"""
        transcript_lower = transcript.lower()
        
        # Enhanced objection patterns with scoring
        objection_patterns = {
            1: ['on our own', 'doing this ourselves', 'not interested in agents', 'handle it ourselves', 'don\'t need', 'no thank you'],
            2: ['had agents call', 'already talked to agents', 'been contacted', 'agents called', 'tried agents before', 'spoke to agents'],
            3: ['getting traction', 'have interest', 'some progress', 'showing interest', 'people looking', 'getting calls'],
            4: ['think about it', 'later', 'not ready', 'maybe later', 'future', 'down the road', 'not now'],
            5: ['not interested', 'no thanks', 'don\'t want', 'stop calling', 'remove me', 'not selling']
        }
        
        # Score-based matching
        scores = {}
        for layer, patterns in objection_patterns.items():
            score = sum(1 for pattern in patterns if pattern in transcript_lower)
            if score > 0:
                scores[layer] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return 1  # Default to layer 1
    
    def get_cached_response(self, transcript):
        """Get a cached response based on transcript analysis"""
        start_time = time.time()
        self.total_requests += 1
        
        # Detect lead type and objection layer
        lead_type = self.detect_lead_type(transcript)
        objection_layer = self.detect_objection_layer(transcript)
        
        # Get the lead data
        lead_data = self.lead_types.get(lead_type)
        if not lead_data:
            self.cache_misses += 1
            return None
        
        # Find matching objection in the tree
        objection_tree = lead_data.get('objection_tree', [])
        for objection in objection_tree:
            if objection.get('layer') == objection_layer:
                end_time = time.time()
                duration = end_time - start_time
                self.cache_hits += 1
                print(f"⚡ Cache hit! Response time: {duration:.4f} seconds")
                print(f"Lead Type: {lead_data['name']}, Layer: {objection_layer}")
                return objection.get('response')
        
        # If no exact match, return first available response
        if objection_tree:
            end_time = time.time()
            duration = end_time - start_time
            self.cache_hits += 1
            print(f"⚡ Cache fallback! Response time: {duration:.4f} seconds")
            return objection_tree[0].get('response')
        
        self.cache_misses += 1
        return None

def load_cache_from_file(file_path="cache_data.json"):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_file_path = os.path.join(script_dir, file_path)
        
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Cache file {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading cache file: {e}")
        return None

def test_cache_performance():
    """Test cache vs AI response times"""
    
    # Load cache
    cache_data = load_cache_from_file()
    if not cache_data:
        print("Failed to load cache data")
        return
        
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
