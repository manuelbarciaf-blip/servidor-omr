import os
import time
import sys
import json
import tempfile
import subprocess
from datetime import datetime
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# ───────────────────────────────────────────────────────────────
# DEPENDENCIAS
# ───────────────────────────────────────────────────────────────
try:
    import requests
    from web3 import Web3
except ImportError:
    print("❌ Instala dependencias: pip install web3 requests")
    sys.exit(1)

# ───────────────────────────────────────────────────────────────
# CONFIG BOT IONIC
# ───────────────────────────────────────────────────────────────

CHECK_INTERVAL = 30
LIQUIDITY_THRESHOLD = 20  # USD mínimo para alertar

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

RPC_URLS = [
    "https://rpc.api.lisk.com",
    "https://lisk.drpc.org",
    "https://1135.rpc.thirdweb.com",
]

MARKETS = {
    "USDC": Web3.to_checksum_address("0x7682C12F6D1af845479649c77A9E7729F0180D78"),
    "USDT": Web3.to_checksum_address("0x0D72f18BC4b4A2F0370Af6D799045595d806636F"),
}

UNDERLYING = {
    "USDC": {"decimals": 6},
    "USDT": {"decimals": 6},
}

BLOCKS_PER_YEAR = 15_768_000

ABI_CTOKEN = [
    {"name": "getCash", "type": "function", "inputs": [], "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "totalBorrows", "type": "function", "inputs": [], "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "supplyRatePerBlock", "type": "function", "inputs": [], "outputs": [{"type": "uint256"}], "stateMutability": "view"},
]

# ───────────────────────────────────────────────────────────────
# FUNCIONES BOT
# ───────────────────────────────────────────────────────────────

def connect():
    for url in RPC_URLS:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 10}))
            if w3.is_connected():
                return w3
        except:
            continue
    raise Exception("❌ No se pudo conectar al RPC Lisk")

def rate_to_apy(rate):
    r = rate / 1e18
    return round(((1 + r) ** BLOCKS_PER_YEAR - 1) * 100, 2)

def get_liquidity(w3, symbol):
    ct = w3.eth.contract(address=MARKETS[symbol], abi=ABI_CTOKEN)
    dec = UNDERLYING[symbol]["decimals"]
    div = 10 ** dec

    cash    = ct.functions.getCash().call()
    borrows = ct.functions.totalBorrows().call()
    s_rate  = ct.functions.supplyRatePerBlock().call()

    tvl = cash + borrows
    util = round((borrows / tvl * 100), 2) if tvl > 0 else 0

    return {
        "symbol": symbol,
        "cash": round(cash / div, 2),
        "tvl": round(tvl / div, 2),
        "utilization_pct": util,
        "supply_apy_pct": rate_to_apy(s_rate),
    }

def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram no configurado")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"})
    return r.status_code == 200

def format_alert(data):
    return (
        f"🚨 <b>Liquidez disponible en {data['symbol']}</b>\n"
        f"💵 Cash: ${data['cash']}\n"
        f"📊 TVL: ${data['tvl']}\n"
        f"📈 APY: {data['supply_apy_pct']}%\n"
    )

last_alert = {"USDC": 0, "USDT": 0}

def check_and_alert(w3):
    now = time.time()
    for symbol in ["USDC", "USDT"]:
        try:
            data = get_liquidity(w3, symbol)
            cash = data["cash"]

            if cash >= LIQUIDITY_THRESHOLD:
                if now - last_alert[symbol] > 300:
                    print(f"🔔 {symbol}: ${cash} → alerta enviada")
                    send_telegram(format_alert(data))
                    last_alert[symbol] = now
                else:
                    print(f"⏳ {symbol}: alerta reciente")
            else:
                print(f"💤 {symbol}: ${cash} (< {LIQUIDITY_THRESHOLD})")

        except Exception as e:
            print(f"⚠️ Error {symbol}: {e}")

# ───────────────────────────────────────────────────────────────
# SERVIDOR HTTP PARA OMR
# ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/health"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OMR + Ionic Bot OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/corregir_omr":
            self.send_response(404)
            self.end_headers()
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(body)
                ruta = tmp.name

            try:
                output = subprocess.check_output(["python3", "omr_local.py", ruta], stderr=subprocess.STDOUT).decode()
                data = json.loads(output)
            except Exception as e:
                data = {"ok": False, "error": str(e)}

            os.remove(ruta)

            resp = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        except Exception as e:
            resp = json.dumps({"ok": False, "error": f"Excepción: {e}"}).encode()
            self.send_response(500)
            self.end_headers()
            self.wfile.write(resp)

    def log_message(self, *args):
        return

# ───────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────

def start_http_server():
    PORT = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"🌐 Servidor HTTP en puerto {PORT}")
    server.serve_forever()

def start_liquidity_bot():
    print("🔗 Conectando RPC Lisk…")
    w3 = connect()
    print("✅ RPC conectado\n")

    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking liquidity…")
        check_and_alert(w3)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    threading.Thread(target=start_http_server, daemon=True).start()
    start_liquidity_bot()
