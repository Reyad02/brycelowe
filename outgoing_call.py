import base64
import json
import os
import time
import threading

from flask import Flask, request, Response
from flask_sock import Sock
import ngrok
from twilio.rest import Client
from dotenv import load_dotenv

from twilio_transcriber import TwilioTranscriber

load_dotenv()

PORT = 5000
DEBUG = False
VOICE_ROUTE = '/voice'
WEBSOCKET_ROUTE = '/realtime'

# Twilio credentials
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
api_key = os.getenv('TWILIO_API_KEY_SID')
api_secret = os.getenv('TWILIO_API_SECRET')
TWILIO_NUMBER = os.getenv('TWILIO_NUMBER')
TARGET_NUMBER = os.getenv('TARGET_NUMBER')
client = Client(api_key, api_secret, account_sid)

# ngrok authentication
# ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN"))

# Flask + WebSocket app
app = Flask(__name__)
sock = Sock(app)

@app.route(VOICE_ROUTE, methods=["POST"])
def voice_response():
    xml = f"""
<Response>
    <Say>Speak now. Your voice will be transcribed.</Say>
    <Connect>
        <Stream url="wss://a80f566262ce.ngrok-free.app{WEBSOCKET_ROUTE}" />
    </Connect>
</Response>
""".strip()
    return Response(xml, mimetype='text/xml')

# @sock.route(WEBSOCKET_ROUTE)
# def transcription_socket(ws):
#     transcriber = TwilioTranscriber()
#     while True:
#         data = json.loads(ws.receive())
#         match data["event"]:
#             case "connected":
#                 transcriber.connect()
#                 print("Transcriber connected")
#             case "start":
#                 print("Call started")
#             case "media":
#                 payload_b64 = data["media"]["payload"]
#                 # print(f"Media received: {payload_b64} ")
#                 # print(f"\n Media received: {len(payload_b64)} bytes \n")
#                 payload_mulaw = base64.b64decode(payload_b64)
#                 # print("Media received", len(payload_mulaw), "bytes", payload_mulaw[:20])
#                 transcriber.stream(payload_mulaw)
#             case "stop":
#                 print("Call ended")
#                 transcriber.close()
#                 break


@sock.route(WEBSOCKET_ROUTE)
def transcription_socket(ws):
    transcriber = TwilioTranscriber()
    transcriber.connect()

    while True:
        data = json.loads(ws.receive())

        match data["event"]:
            case "connected":
                print("Transcriber connected")
            case "start":
                print("Call started")
            case "media":
                payload_b64 = data["media"]["payload"]
                payload_bytes = base64.b64decode(payload_b64)
                transcriber.stream(payload_bytes)
            case "stop":
                print("Call ended")
                transcriber.close()
                break


def run_flask():
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG, use_reloader=False)

if __name__ == "__main__":
    try:
        # Start ngrok tunnel
        # listener = ngrok.forward(f"http://localhost:{PORT}")
        NGROK_URL = "https://a80f566262ce.ngrok-free.app"
        print(f"Ngrok tunnel running at: {NGROK_URL}")

        # Start Flask app in a background thread
        threading.Thread(target=run_flask).start()
        time.sleep(2)  # let Flask boot

        # Make outbound call
        print(f"Dialing {TARGET_NUMBER} from {TWILIO_NUMBER}...")
        call = client.calls.create(
            from_=TWILIO_NUMBER,
            to=TARGET_NUMBER,
            url=f"{NGROK_URL}{VOICE_ROUTE}"
        )
        print(f"Call initiated. SID: {call.sid}")

        input("Press Enter to stop server...\n")

    finally:
        ngrok.disconnect()

