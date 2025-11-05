import qbittorrentapi, os, requests

token = os.getenv('token')
res = requests.get("https://raw.githubusercontent.com/piyushpradhan22/credentials/refs/heads/main/credentials.json",
                   headers={"Authorization" : f"token {token}"}).json()

username = res['username']
password = res['password']

# Connection details
conn_info = {'host': 'localhost', 'port': 3333, 'username': username, 'password': password}

# Instantiate the client
qbt_client = qbittorrentapi.Client(**conn_info)
qbt_client.auth_log_in()

## Install Search Plugins
urls = ["https://raw.githubusercontent.com/LightDestory/qBittorrent-Search-Plugins/master/src/engines/snowfl.py",
        "https://raw.githubusercontent.com/BurningMop/qBittorrent-Search-Plugins/refs/heads/main/therarbg.py"]
qbt_client.search_install_plugin(sources=urls)

print("Search Plugins installed")
