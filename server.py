"""
═══════════════════════════════════════════════════════════════════════════
  IONIC MONEY — Bot d'alertes Telegram (Render.com 24/7)
═══════════════════════════════════════════════════════════════════════════

Bot qui surveille en continu la liquidité USDC.e & USDT sur Ionic Money (Lisk)
et envoie des alertes Telegram dès qu'il y a ≥ $20 de liquidité.

Optimisé pour tourner 24/7 sur Render.com (plan gratuit).
═══════════════════════════════════════════════════════════════════════════
"""

import time
import sys
from datetime import datetime

try:
    import requests
    from web3 import Web3
except ImportError:
    print("❌ Dépendances manquantes. Installez : pip install web3 requests")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TELEGRAM_BOT_TOKEN = "8591638141:AAHwunx9OV-5TV-2ThKaWUmXgX2rXL0kZWU"
TELEGRAM_CHAT_ID   = "862574975"

# Seuil minimum pour alerter
LIQUIDITY_THRESHOLD = 20  # dollars

# Intervalle entre chaque vérification
CHECK_INTERVAL = 30  # secondes

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

# ══════════════════════════════════════════════════════════════════════════════
#  ABIs
# ══════════════════════════════════════════════════════════════════════════════

ABI_CTOKEN = [
    {"name": "getCash",            "type": "function", "inputs": [], "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "totalBorrows",       "type": "function", "inputs": [], "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "supplyRatePerBlock", "type": "function", "inputs": [], "outputs": [{"type": "uint256"}], "stateMutability": "view"},
]

# ══════════════════════════════════════════════════════════════════════════════
#  WEB3
# ══════════════════════════════════════════════════════════════════════════════

def connect():
    """Se connecte au RPC Lisk."""
    for url in RPC_URLS:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 10}))
            if w3.is_connected():
                return w3
        except Exception:
            continue
    raise ConnectionError("Impossible de se connecter au RPC Lisk")


def rate_to_apy(rate):
    """Convertit un taux par bloc en APY annuel."""
    r = rate / 1e18
    return round(((1 + r) ** BLOCKS_PER_YEAR - 1) * 100, 2)


def get_liquidity(w3, symbol):
    """Récupère les données de liquidité pour un marché."""
    ctoken = MARKETS[symbol]
    dec    = UNDERLYING[symbol]["decimals"]
    div    = 10 ** dec

    ct = w3.eth.contract(address=ctoken, abi=ABI_CTOKEN)

    cash    = ct.functions.getCash().call()
    borrows = ct.functions.totalBorrows().call()
    s_rate  = ct.functions.supplyRatePerBlock().call()

    tvl  = cash + borrows
    util = round((borrows / tvl * 100), 2) if tvl > 0 else 0.0

    return {
        "symbol":          symbol,
        "cash":            round(cash / div, 2),
        "tvl":             round(tvl / div, 2),
        "utilization_pct": util,
        "supply_apy_pct":  rate_to_apy(s_rate),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════

def send_telegram(message):
    """Envoie un message Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"⚠️  Erreur Telegram : {e}")
        return False


def format_alert(data):
    """Formate un message d'alerte."""
    sym  = data["symbol"]
    cash = data["cash"]
    tvl  = data["tvl"]
    util = data["utilization_pct"]
    sapy = data["supply_apy_pct"]

    icon = "💧" if cash > 100 else "🚨"

    return f"""
{icon} <b>IONIC MONEY — Liquidité {sym}</b>

💰 <b>Liquidité dispo :</b> ${cash:,.2f}
🏦 <b>TVL totale :</b> ${tvl:,.2f}
📊 <b>Utilisation :</b> {util}%
📈 <b>APY Dépôt :</b> {sapy}%

🕐 {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC
    """.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  MONITORING
# ══════════════════════════════════════════════════════════════════════════════

last_alert = {"USDC": 0, "USDT": 0}


def check_and_alert(w3):
    """Vérifie la liquidité et envoie une alerte si nécessaire."""
    global last_alert
    now = time.time()

    for symbol in ["USDC", "USDT"]:
        try:
            data = get_liquidity(w3, symbol)
            cash = data["cash"]

            # Alerte si liquidité ≥ seuil
            if cash >= LIQUIDITY_THRESHOLD:
                # Throttle : max 1 alerte toutes les 5 minutes pour éviter le spam
                if (now - last_alert[symbol]) > 300:  # 5 minutes
                    print(f"✅ {symbol} : ${cash:,.2f} → Alerte envoyée")
                    if send_telegram(format_alert(data)):
                        last_alert[symbol] = now
                    else:
                        print(f"   ⚠️  Échec envoi Telegram")
                else:
                    elapsed = int(now - last_alert[symbol])
                    print(f"⏳ {symbol} : ${cash:,.2f} (dernière alerte il y a {elapsed}s)")
            else:
                print(f"💤 {symbol} : ${cash:,.2f} (< ${LIQUIDITY_THRESHOLD})")

        except Exception as e:
            print(f"⚠️  Erreur {symbol} : {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 70)
    print("  🤖 IONIC LIQUIDITY ALERT BOT — Render.com")
    print("═" * 70)
    print(f"  📡 Réseau      : Lisk (Chain 1135)")
    print(f"  💰 Seuil       : ${LIQUIDITY_THRESHOLD}")
    print(f"  ⏱️  Intervalle  : {CHECK_INTERVAL}s")
    print(f"  🔕 Anti-spam   : Max 1 alerte / 5 min par token")
    print(f"  💬 Chat ID     : {TELEGRAM_CHAT_ID}")
    print("═" * 70)
    print()

    # Test Telegram
    print("📤 Test de connexion Telegram...")
    if send_telegram("🤖 <b>Bot Ionic Money démarré</b>\n\nSurveillance USDC.e & USDT activée sur Render.com."):
        print("✅ Message de test envoyé\n")
    else:
        print("⚠️  Échec du test Telegram\n")

    # Connexion RPC
    print("🔗 Connexion au RPC Lisk...")
    w3 = connect()
    print(f"✅ Connecté — Chain ID: {w3.eth.chain_id}\n")
    
    # RENDER.COM FIX: Démarrer serveur HTTP pour port binding
    print("🌐 Démarrage serveur HTTP pour Render.com...")
    PORT = int(os.environ.get('PORT', 10000))
    
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading
    
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Ionic Money Bot - Running OK')
        
        def log_message(self, format, *args):
            pass  # Silence les logs HTTP
    
    server = HTTPServer(('0.0.0.0', PORT), HealthHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"✅ Serveur HTTP démarré sur port {PORT}\n")

    print(f"👁️  Surveillance démarrée (vérification toutes les {CHECK_INTERVAL}s)...\n")

    try:
        while True:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] Vérification...")
            check_and_alert(w3)
            print()
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n👋 Arrêté.")
        send_telegram("🛑 <b>Bot Ionic Money arrêté</b>")


if __name__ == "__main__":
    main()
