"""Exit 0 if the guidance endpoint answers a one-token request, 1 otherwise.

Used by run_tcad_resume_llm.sh to wait for the endpoint after an outage (or
an exhausted account balance) without starting guided runs that the outage
rule would only abandon.
"""
import json
import os
import sys
import urllib.request

req = urllib.request.Request(
    "https://api.deepseek.com/chat/completions",
    data=json.dumps({"model": os.environ.get("APPROXLG_MODEL",
                                             "deepseek-v4-flash"),
                     "messages": [{"role": "user", "content": "Reply OK"}],
                     "max_tokens": 2}).encode(),
    headers={"Authorization": "Bearer " + os.environ.get("DEEPSEEK_API_KEY", ""),
             "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception as e:                                  # noqa: BLE001
    print(f"endpoint not available: {e}", file=sys.stderr)
    sys.exit(1)
