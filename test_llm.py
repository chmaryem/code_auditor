"""Quick LLM key + invoke test"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Load .env
env = Path(__file__).parent / ".env"
for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

# Check keys
gr  = os.environ.get("GOOGLE_API_KEY", "")
or_ = os.environ.get("OPENROUTER_API_KEY", "")
print(f"GOOGLE_API_KEY     : {gr[:8]+'...' if gr else 'MANQUANT'}")
print(f"OPENROUTER_API_KEY : {or_[:8]+'...' if or_ else 'MANQUANT'}")

if not gr and not or_:
    print("\nAucune cle LLM — impossible de faire une analyse IA")
    sys.exit(1)

# Quick invoke test
print("\nTest invoke_with_fallback...")
from services.llm_factory import invoke_with_fallback
result = invoke_with_fallback("Say hello in one sentence.", label="ci-test")
if result:
    print(f"LLM OK: {result[:200]}")
else:
    print("LLM: None returned — tous les providers ont echoue")
