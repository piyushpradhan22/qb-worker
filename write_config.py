import os, requests

token = os.getenv('token')
res = requests.get("https://raw.githubusercontent.com/piyushpradhan22/credentials/refs/heads/main/credentials.json",
                   headers={"Authorization" : f"token {token}"}).json()

username = res['username']
enc_password = res['enc_password']

# Write qBittorrent.conf with environment variables
cred = f"""[Preferences]
WebUI\\Username={username}
WebUI\\Password_PBKDF2={enc_password}

[AutoRun]
enabled=true
program=python3 copy_tor.py
"""

with open("/home/user/.config/qBittorrent/qBittorrent.conf", "w") as f:
    f.write(cred)