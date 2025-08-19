import json
import os
import time
import threading
import re
from flask import Flask, request, Response
from flask_sock import Sock
import ngrok
from twilio.rest import Client
from dotenv import load_dotenv
from openai import OpenAI
import time
import google.generativeai as genai
from collections import OrderedDict
from datetime import datetime, timedelta


load_dotenv()

client_of_AI = OpenAI() 
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gemini_model = genai.GenerativeModel("gemini-2.5-flash")  
# Cache system for fast response lookup
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
    
    def add_to_cache(self, transcript, response, lead_type, layer=None):
        """Add a new response to cache for future use"""
        cache_key = f"{lead_type or 'unknown'}_{layer or 1}_{hash(transcript) % 1000}"
        self.cache[cache_key] = {
            'transcript': transcript,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'lead_type': lead_type,
            'layer': layer
        }
        
        # Implement LRU eviction
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove oldest item
    
    def get_cached_response(self, transcript, lead_type):
        """Get a cached response based on transcript analysis"""
        start_time = time.time()
        self.total_requests += 1
        
        # Detect lead type and objection layer
        # lead_type = self.detect_lead_type(transcript)
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
            # self.cache_hits += 1
            self.cache_misses += 1
            print(f"⚡ Cache fallback! Response time: {duration:.4f} seconds")
            # return objection_tree[0].get('response')
            return None
        
        self.cache_misses += 1
        return None

# Load cache data from file
def load_cache_from_file(file_path="cache_data.json"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file_path = os.path.join(script_dir, file_path)
    
    with open(cache_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Initialize cache with data from file
cache_data = load_cache_from_file()
response_cache = ResponseCache(cache_data)  

PORT = 5000
DEBUG = True
VOICE_ROUTE = '/voice'

# Twilio credentials
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
api_key = os.getenv('TWILIO_API_KEY_SID')
api_secret = os.getenv('TWILIO_API_SECRET')
TWILIO_NUMBER = os.getenv('TWILIO_NUMBER')
TARGET_NUMBER = os.getenv('TARGET_NUMBER')
SECOND_NUMBER = os.getenv('TARGET_NUMBER_1')
client = Client(api_key, api_secret, account_sid)

app = Flask(__name__)
sock = Sock(app)

lead_transcript = []
trust_score = 0.0
lock = threading.Lock()

@app.route(VOICE_ROUTE, methods=["POST"])
def voice_response():
    xml = f"""
<Response>
    <Say>Dialing the second number now...</Say>
    <Start>
        <Transcription statusCallbackUrl="{NGROK_URL}/transcription-webhook" track="both_tracks"/>
    </Start>
    <Dial>
        <Number>{SECOND_NUMBER}</Number>
    </Dial>
</Response>

""".strip()
    return Response(xml, mimetype='text/xml')

@app.route('/transcription-webhook', methods=['POST'])
def transcription_webhook():
    data = request.form.to_dict()
    transcription_data = data.get('TranscriptionData')
    track = data.get('Track')
    
    if transcription_data:
        td = json.loads(transcription_data)
        transcript = td.get('transcript')
        confidence = td.get('confidence')

        if track == 'inbound_track':
            speaker = "Agent"
            print(f"{speaker}: {transcript} (Confidence: {confidence})")

        elif track == 'outbound_track':
            speaker = "Lead"
            print(f"{speaker}: {transcript} (Confidence: {confidence})")
            
            
            manual_lead_type = "L001"  


            # Try cache first for ultra-fast response
            cached_response = response_cache.get_cached_response(transcript,manual_lead_type)
            
            if cached_response:
                print(f"⚡ CACHED reply to Agent: {cached_response}")
            else:
                # Fallback to AI if no cache hit
                print("🔄 Cache miss, falling back to AI...")
                suggestion_from_gemini = get_response_suggestion_gemini(transcript)
                print(f"🤖 AI reply to Agent from gemini: {suggestion_from_gemini}")
                
                # Optionally add AI response to cache for future use
                lead_type = response_cache.detect_lead_type(transcript)
                layer = response_cache.detect_objection_layer(transcript)
                response_cache.add_to_cache(transcript, suggestion_from_gemini, lead_type, layer)
            

        else:
            speaker = "Unknown"
            print(f"{speaker}: {transcript} (Confidence: {confidence})")

    return ('', 204)


@app.route('/cache-stats', methods=['GET'])
def cache_stats():
    """Get cache performance statistics"""
    stats = response_cache.get_cache_stats()
    return json.dumps(stats, indent=2), 200, {'Content-Type': 'application/json'}


def get_trust_score_from_llm(text):
    """
    Sends conversation chunk to LLM to get a trust score between 0–100.
    """
    prompt = f"""
    You are an AI call analyzer.
    Based on the following lead's speech, give a trust score between 0 and 100.
    Higher means they are open, honest, and likely cooperative.

    Lead's speech:
    {text}

    Respond with only a number.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        score_str = resp.choices[0].message.content.strip()
        return float(score_str)
    except Exception as e:
        print(f"LLM trust score error: {e}")
        return None
    


def trust_score_updater():
    global trust_score
    while True:
        time.sleep(15)  # every 15 seconds
        with lock:
            if not lead_transcript:
                continue
            text_chunk = " ".join(lead_transcript)
            score = get_trust_score_from_llm(text_chunk)
            if score is not None:
                trust_score = score
                print(f"[Trust Score Updated] {trust_score}")
            lead_transcript.clear()  # clear after processing



def get_response_suggestion(agent_transcript):
    try:
        start_time = time.time()
        response = client_of_AI.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant. Based on what the agent just said, suggest what the agent should say next to the customer."},
                {"role": "user", "content": agent_transcript}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        end_time = time.time() 
        duration = end_time - start_time
        
        print(f"👍Response time from openai: {duration:.2f} seconds")

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "[Error getting suggestion]"
    
def get_response_suggestion_gemini(agent_transcript):
    try:
        start_time = time.time()

        # Generate content
        response = gemini_model.generate_content(
            f"Suggest a short, simple sentence the agent should say next. User said: {agent_transcript}",
            generation_config=genai.types.GenerationConfig(
                # max_output_tokens=100,     
                temperature=0.7,          
            )
        )

        end_time = time.time()
        duration = end_time - start_time
        print(f"❤️Response time from the gemini: {duration:.2f} seconds")
        print(f"\nGemini Response: {response}\n")


        return response.text.strip()

    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return "[Error getting suggestion]"
    
def analyze_agent(agent_transcript):
    try:
        start_time = time.time()

        # Generate content
        response = gemini_model.generate_content(
            """You are a human behavioral analyzer. Your job is to analyze an agent's call performance based on the agent responds. Focus on:

1. **Behavioral Events**: Note specific actions the agent takes,  and the category (e.g., opening, rapport, objection handling, closing).
2. **Closing Activity**: Track whether a call-to-action (CTA) was issued, whether an appointment attempt was made, and guess the call outcome (success/failure/pending).
3. **Tone Samples**: Identify the agent’s tone (warm, confident, neutral, etc.) and energy (low, neutral, high).
4. **Adaptability Moments**: Identify when the agent changed strategy, pacing, or phrasing to match the user’s behavior.
5. **Coaching Tags Raw**: Tag behaviors like reassuring, assertive, empathetic, etc., with timestamps.
6. **Links Library**: Reference any helpful external resources for the behaviors noted.

Your output **MUST be JSON** following this structure:

{
  "behavioral_events": [
    {"agent_action":"action description","ai_prompt_used":true/false,"category":"category name"}
  ],
  "closing_activity": {"cta_issued":true/false,"appointment_attempted":true/false,"status_guess":"success/failure/pending"},
  "tone_samples": [{"tone":"tone label","energy":"low/neutral/high"}],
  "adaptability_moments": [{"type":"change type","note":"description"}],
  "coaching_tags_raw": [{"tag":"tag name"}],
  "links_library": [{"name":"resource title","url":"link","topic":"topic"}]
}

Analyze the following transcript and fill in all relevant fields:

User said: {agent_transcript}
""",
            generation_config=genai.types.GenerationConfig(
                # max_output_tokens=100,     
                temperature=0.7,          
            )
        )

        end_time = time.time()
        duration = end_time - start_time
        print(f"❤️Response time from the gemini: {duration:.2f} seconds")
        print(f"\nGemini Response: {response}\n")


        return response.text.strip()

    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return "[Error getting suggestion]"

def run_flask():
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG, use_reloader=False)

if __name__ == "__main__":
    try:
        # Print cache initialization info
        stats = response_cache.get_cache_stats()
        print(f"🚀 Cache system initialized with {stats['lead_types_loaded']} lead types")
        
        # listener = ngrok.forward(PORT, authtoken_from_env=True, proto="http,https,tcp")
        NGROK_URL = "https://bfd11e9c127e.ngrok-free.app"
        print(f"Ngrok tunnel running at: {NGROK_URL}")
        print(f"📊 Cache stats endpoint: {NGROK_URL}/cache-stats")

        flask_thread = threading.Thread(target=run_flask)
        flask_thread.daemon = True
        flask_thread.start()
        time.sleep(2)

        print(f"Dialing {TARGET_NUMBER} from {TWILIO_NUMBER}...")
        call = client.calls.create(
            from_=TWILIO_NUMBER,
            to=TARGET_NUMBER,
            url=f"{NGROK_URL}{VOICE_ROUTE}"
        )
        print(f"Call initiated. SID: {call.sid}")

        # Print cache stats every 30 seconds
        start_time = time.time()
        while True:
            time.sleep(30)
            current_time = time.time()
            if current_time - start_time >= 30:
                stats = response_cache.get_cache_stats()
                print(f"\n📊 Cache Performance: Hit Rate: {stats['hit_rate']}, Total Requests: {stats['total_requests']}")
                start_time = current_time

    except KeyboardInterrupt:
        print("Shutting down...")
        stats = response_cache.get_cache_stats()
        print(f"\n📊 Final Cache Stats: {stats}")
    finally:
        ngrok.disconnect()