import pyaudio
import websocket
import json
import threading
import time
import wave
from urllib.parse import urlencode
from datetime import datetime
from google import genai
import uuid
import hashlib
import os

from dotenv import load_dotenv


load_dotenv()


print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))

CONNECTION_PARAMS = {
    "sample_rate": 16000,
    "format_turns": True,
}
API_ENDPOINT_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
API_ENDPOINT = f"{API_ENDPOINT_BASE_URL}?{urlencode(CONNECTION_PARAMS)}"

# Audio Configuration
FRAMES_PER_BUFFER = 800
SAMPLE_RATE = CONNECTION_PARAMS["sample_rate"]
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Global variables
audio = None
stream = None
ws_app = None
audio_thread = None
stop_event = threading.Event()
recorded_frames = []
recording_lock = threading.Lock()

# GenAI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Calibration test configuration
TEST_PROMPTS = [
    {
        "id": "C001",
        "scenario": "FSBO-Fee-Objection",
        "difficulty": "medium",
        "prompt": "Respond to: 'I'm worried about fees.'",
        "expected_tone": "empathetic",
        "word_limit": 25,
        "emphasis": "reassurance"
    },
    {
        "id": "C002",
        "scenario": "Price-Objection",
        "difficulty": "medium",
        "prompt": "Respond to: 'Your price seems too high.'",
        "expected_tone": "confident",
        "word_limit": 30,
        "emphasis": "value-proposition"
    },
]

# Calibration state
calibration_state = {
    "current_prompt_index": 0,
    "responses": [],
    "start_time": None,
    "user_id": str(uuid.uuid4()),
    "test_prompts": TEST_PROMPTS
}

def evaluate_response(scenario, expected_tone, word_limit, emphasis, agent_response, lead_saying):
    prompt = f"""
    Evaluate the agent's response based on the following criteria:

    Scenario: {scenario}
    Expected Tone: {expected_tone}
    Word Limit: {word_limit}
    Emphasis: {emphasis}
    Lead Say: {lead_saying}
    Agent's Response: "{agent_response}"

    Scoring Criteria:
    - Sentiment Score (0.0 to 1.0): How well the response matches the required sentiment.
    - Tone Score (0.0 to 1.0): How well the response matches the expected tone.
    - Pacing Score (0.0 to 1.0): How well the response manages pacing within the word limit and emphasis.

    Output only a valid JSON object in the following format:
    {{
        "sentiment_score": float,
        "tone_score": float,
        "pacing_score": float,
    }}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    
    try:
        scores_json = json.loads(response.text)
        return scores_json
    except:
        print("Failed to parse Gemini response")
        return None

def calculate_final_calibration():
    """Calculate the final calibration results based on all responses"""
    if not calibration_state["responses"]:
        return None
    
    # Calculate average scores
    total_responses = len(calibration_state["responses"])
    avg_sentiment = sum(r["sentiment_score"] for r in calibration_state["responses"]) / total_responses
    avg_tone = sum(r["tone_score"] for r in calibration_state["responses"]) / total_responses
    avg_pacing = sum(r["pacing_score"] for r in calibration_state["responses"]) / total_responses
    
    # Calculate final score (0-100)
    final_score = int((avg_sentiment + avg_tone + avg_pacing) / 3 * 100)
    
    # Determine recommended tier based on scores
    if avg_tone >= 0.8:
        tone_category = "Empathy" if "empath" in calibration_state["responses"][0]["expected_tone"].lower() else "Clarity"
    else:
        tone_category = "Clarity"
    
    if avg_pacing >= 0.8:
        pacing_category = "High-Pacing"
    elif avg_pacing >= 0.5:
        pacing_category = "Medium-Pacing"
    else:
        pacing_category = "Low-Pacing"
    
    recommended_tier = f"{tone_category}-{pacing_category}"
    
    # Create calibration hash
    hash_input = f"{calibration_state['user_id']}{final_score}{recommended_tier}"
    calibration_hash = hashlib.sha256(hash_input.encode()).hexdigest()
    
    # Prepare final result
    end_time = datetime.now().isoformat()
    duration_seconds = (datetime.fromisoformat(end_time) - 
                       datetime.fromisoformat(calibration_state["start_time"])).total_seconds()
    
    calibration_result = {
        "calibration_engine": {
            "user_id": calibration_state["user_id"],
            "test_prompts": calibration_state["test_prompts"],
            "user_responses": calibration_state["responses"],
            "calibration_hash": calibration_hash
        },
        "calibration_result": {
            "user_id": calibration_state["user_id"],
            "date": end_time,
            "duration_seconds": duration_seconds,
            "overall_score": final_score,
            "tier": recommended_tier,
            "summary": "Calibration complete! All live prompts will now match your natural rhythm.",
            "display_message": "You're calibrated! Expect AI prompts that feel like you."
        }
    }
    
    return calibration_result

def display_calibration_summary(calibration_result):
    """Display the final calibration summary to the user"""
    if not calibration_result:
        print("No calibration results to display")
        return
    
    result = calibration_result["calibration_result"]
    print("\n" + "="*50)
    print("Calibration Complete!")
    print("="*50)
    print(f"Your Personalized Style: {result['tier']}")
    print(f"Score: {result['overall_score']}/100")
    print(f"\n{result['summary']}")
    print("\n" + "="*50)
    
    # Optionally save the full results to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"calibration_results_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(calibration_result, f, indent=2)
    print(f"\nFull calibration results saved to: {filename}")

def on_open(ws):
    print("WebSocket connection opened.")
    print(f"Connected to: {API_ENDPOINT}")
    
    # Initialize calibration
    calibration_state["start_time"] = datetime.now().isoformat()
    print("\nStarting calibration test...")
    print(f"Prompt 1 of {len(TEST_PROMPTS)}: {TEST_PROMPTS[0]['prompt']}")
    
    def stream_audio():
        global stream
        print("\nSpeak your response now...")
        while not stop_event.is_set():
            try:
                audio_data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                with recording_lock:
                    recorded_frames.append(audio_data)
                ws.send(audio_data, websocket.ABNF.OPCODE_BINARY)
            except Exception as e:
                print(f"Error streaming audio: {e}")
                break
        print("Audio streaming stopped.")

    global audio_thread
    audio_thread = threading.Thread(target=stream_audio)
    audio_thread.daemon = True
    audio_thread.start()

def on_message(ws, message):
    try:
        data = json.loads(message)
        msg_type = data.get('type')

        if msg_type == "Begin":
            session_id = data.get('id')
            expires_at = data.get('expires_at')
            print(f"\nSession began: ID={session_id}, ExpiresAt={datetime.fromtimestamp(expires_at)}")
        elif msg_type == "Turn":
            transcript = data.get('transcript', '')
            formatted = data.get('turn_is_formatted', False)
            
            if formatted:
                print('\r' + ' ' * 80 + '\r', end='')
                print(f"You said: {transcript}")
                
                # Get current prompt
                current_prompt = TEST_PROMPTS[calibration_state["current_prompt_index"]]
                
                # Evaluate response
                scores = evaluate_response(
                    scenario=current_prompt["scenario"],
                    expected_tone=current_prompt["expected_tone"],
                    word_limit=current_prompt["word_limit"],
                    emphasis=current_prompt["emphasis"],
                    agent_response=transcript,
                    lead_saying=current_prompt["prompt"]
                )
                
                if scores:
                    # Store response with scores
                    response_data = {
                        "text": transcript,
                        "sentiment_score": scores["sentiment_score"],
                        "tone_score": scores["tone_score"],
                        "pacing_score": scores["pacing_score"],
                        "expected_tone": current_prompt["expected_tone"],
                        "emphasis": current_prompt["emphasis"]
                    }
                    calibration_state["responses"].append(response_data)
                    
                    # Move to next prompt or finish
                    calibration_state["current_prompt_index"] += 1
                    
                    if calibration_state["current_prompt_index"] < len(TEST_PROMPTS):
                        next_prompt = TEST_PROMPTS[calibration_state["current_prompt_index"]]
                        print(f"\nPrompt {calibration_state['current_prompt_index'] + 1} of {len(TEST_PROMPTS)}: {next_prompt['prompt']}")
                        print("Speak your response...")
                    else:
                        # Calibration complete
                        print("\nAll test prompts completed. Calculating results...")
                        final_result = calculate_final_calibration()
                        display_calibration_summary(final_result)
                        stop_event.set()
                        ws.close()
                else:
                    print("Failed to evaluate response. Please try again.")
                    
        elif msg_type == "Termination":
            audio_duration = data.get('audio_duration_seconds', 0)
            session_duration = data.get('session_duration_seconds', 0)
            print(f"\nSession Terminated: Audio Duration={audio_duration}s, Session Duration={session_duration}s")
    except json.JSONDecodeError as e:
        print(f"Error decoding message: {e}")
    except Exception as e:
        print(f"Error handling message: {e}")

def on_error(ws, error):
    print(f"\nWebSocket Error: {error}")
    stop_event.set()

def on_close(ws, close_status_code, close_msg):
    print(f"\nWebSocket Disconnected: Status={close_status_code}, Msg={close_msg}")
    save_wav_file()
    stop_event.set()

    global stream, audio
    if stream:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        stream = None
    if audio:
        audio.terminate()
        audio = None
    if audio_thread and audio_thread.is_alive():
        audio_thread.join(timeout=1.0)

def save_wav_file():
    if not recorded_frames:
        print("No audio data recorded.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recorded_audio_{timestamp}.wav"

    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            with recording_lock:
                wf.writeframes(b''.join(recorded_frames))
        print(f"Audio saved to: {filename}")
    except Exception as e:
        print(f"Error saving WAV file: {e}")

def run():
    global audio, stream, ws_app

    audio = pyaudio.PyAudio()

    try:
        stream = audio.open(
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
            channels=CHANNELS,
            format=FORMAT,
            rate=SAMPLE_RATE,
        )
        print("Microphone stream opened successfully.")
        print(f"Starting calibration for user: {calibration_state['user_id']}")
    except Exception as e:
        print(f"Error opening microphone stream: {e}")
        if audio:
            audio.terminate()
        return

    ws_app = websocket.WebSocketApp(
        API_ENDPOINT,
        header={"Authorization": os.getenv('ASSEMBLY_AI_API_KEY')},
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws_thread = threading.Thread(target=ws_app.run_forever)
    ws_thread.daemon = True
    ws_thread.start()

    try:
        while ws_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping...")
        stop_event.set()
        if ws_app and ws_app.sock and ws_app.sock.connected:
            try:
                terminate_message = {"type": "Terminate"}
                ws_app.send(json.dumps(terminate_message))
                time.sleep(1)
            except Exception as e:
                print(f"Error sending termination message: {e}")
        if ws_app:
            ws_app.close()
        ws_thread.join(timeout=2.0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        stop_event.set()
        if ws_app:
            ws_app.close()
        ws_thread.join(timeout=2.0)
    finally:
        if stream and stream.is_active():
            stream.stop_stream()
        if stream:
            stream.close()
        if audio:
            audio.terminate()
        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    run()