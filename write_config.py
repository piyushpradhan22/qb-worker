import os, requests

token = os.getenv('token')
res = requests.get("https://raw.githubusercontent.com/piyushpradhan22/credentials/refs/heads/main/credentials.json",
                   headers={"Authorization" : f"token {token}"}).json()

username = res['username']
enc_password = res['enc_password']

# Write qBittorrent.conf with environment variables and reverse proxy support
cred = f"""[Preferences]
WebUI\\Username={username}
WebUI\\Password_PBKDF2={enc_password}
WebUI\\HostHeaderValidation=false
WebUI\\CSRFProtection=false
WebUI\\ReverseProxySupportEnabled=true
WebUI\\TrustedReverseProxiesList=127.0.0.1/32,100.64.0.0/10

[AutoRun]
enabled=true
program=python3 copy_tor_v4.py
"""

with open("/home/user/.config/qBittorrent/qBittorrent.conf", "w") as f:
    f.write(cred)
