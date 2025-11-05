import time, os, qbittorrentapi, re, requests
import pandas as pd
from imdb import Cinemagoer
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from huggingface_hub import HfApi

token = os.getenv('token')
res = requests.get("https://raw.githubusercontent.com/piyushpradhan22/credentials/refs/heads/main/credentials.json",
                   headers={"Authorization" : f"token {token}"}).json()

username = res['username']
password = res['password']

# Connection details
conn_info = {'host': 'localhost', 'port': 7860, 'username': username, 'password': password}

# Instantiate the client
qbt_client = qbittorrentapi.Client(**conn_info)
qbt_client.auth_log_in()

postgres_engine = create_engine(res['postgres_url'], poolclass=NullPool)
ia = Cinemagoer()
hf_token = res['hf_token']

# API instance
api = HfApi(token=hf_token)
repo_id = res['repo_id']
repo_type = "dataset"

def list_video_files(directory, hash):
    video_files = []
    i = 1
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.mkv') or file.endswith('.mp4'):
                file_path = os.path.join(root, file)
                file_name = file_path.split("/")[-1]
                video_files.append({"file_path" : file_path, "size" : os.path.getsize(file_path),
                                    "file_hash" : f"{hash}_{i}" })
                i+=1
    return video_files

# List of characters to be replaced
disallowed_chars = ['—']

def replace_disallowed_chars(folder_name):
    # Create a regex pattern that matches any of the disallowed characters
    pattern = re.compile('|'.join(map(re.escape, disallowed_chars)))
    # Replace matched characters with a dot
    return pattern.sub('.', folder_name)

def parse_torrent_name(name):
    """
    Parse torrent/file name to extract title, season, episode info.
    Returns dict with 'title', 'season' (if found), 'episode' (if found)
    """
    result = {}
    
    # Remove file extension
    name_clean = re.sub(r'\.(mkv|mp4|avi|mov)$', '', name, flags=re.IGNORECASE)
    
    # Season and Episode patterns (in order of specificity)
    # Matches: S01E02, S01 E02, S01.E02, s01e02, Season 1 Episode 2, season 01 episode 02, etc.
    patterns = [
        # S01E02 or S01 E02 or S01.E02 (standard) - must have both S and E markers
        (r'[Ss](\d{1,2})[\s\.\-]*[Ee](\d{1,3})', True),
        # Season 01 Episode 02 (verbose) - full words
        (r'[Ss]eason[\s\.\-]*(\d{1,2})[\s\.\-]*[Ee]pisode[\s\.\-]*(\d{1,3})', True),
        # 1x02 format
        (r'(\d{1,2})x(\d{1,3})', True),
        # Just Episode with number (assume season 1 if only episode found)
        (r'[Ee]pisode[\s\.\-]*(\d{1,3})(?:\D|$)', False),
        (r'[Ee]p[\s\.\-]*(\d{1,3})(?:\D|$)', False),
    ]
    
    season = None
    episode = None
    match_pos = -1
    
    for pattern, has_season in patterns:
        match = re.search(pattern, name_clean, re.IGNORECASE)
        if match:
            match_pos = match.start()
            groups = match.groups()
            if has_season and len(groups) == 2:
                season = int(groups[0])
                episode = int(groups[1])
            elif not has_season and len(groups) == 1:
                # Only episode found, assume season 1
                season = 1
                episode = int(groups[0])
            break
    
    # Extract title (everything before season/episode info or quality tags)
    if match_pos > 0:
        title_part = name_clean[:match_pos]
    else:
        # No season/episode found, try to extract before quality/year info
        quality_match = re.search(r'(720p|1080p|2160p|4k|\d{4}|bluray|webrip|web-dl|hdtv|xvid|x264|x265|hevc)', 
                                  name_clean, re.IGNORECASE)
        if quality_match:
            title_part = name_clean[:quality_match.start()]
        else:
            title_part = name_clean
    
    # Clean up the title
    title = re.sub(r'[\._\-\[\]\(\)]', ' ', title_part)
    title = re.sub(r'\s+', ' ', title).strip()
    
    result['title'] = title if title else name
    
    if season is not None:
        result['season'] = season
    if episode is not None:
        result['episode'] = episode
    
    return result

torrs = [tor for tor in qbt_client.torrents_info() if tor.progress==1 and tor.state != 'pausedUP']
print("Received Torrent" , [tor.name for tor in torrs])
hashes = [tor.hash for tor in torrs]
qbt_client.torrents_pause(hashes)
for tor in torrs:
    #save_path = replace_disallowed_chars(tor['content_path'])
    save_path = tor['content_path']
    if save_path.split(".")[-1] in ['mkv', 'mp4']:
        video_files = [{"file_path" : save_path, "size" : os.path.getsize(save_path),
                                    "file_hash" : tor.hash}]
    else:
        video_files = list_video_files(save_path, tor.hash)
        
    if not video_files:
        print("No video files found...")
    if tor.name[:5] == 'imdb:':
        imdb_id = tor.name.split(":")[1]
        tor.name = tor.name.split(":")[2]
    else:
        imdb_id = None
    for vd in video_files:
        file_path = vd['file_path']
        file_name = file_path.split("/")[-1]
        if 'sample' in file_name.lower():
            continue
        file_hash = vd['file_hash']
        # Upload to HF Dataset
        message = api.upload_file(path_or_fileobj=file_path, path_in_repo=file_hash, repo_id=repo_id, repo_type=repo_type,)
        print(message)
        ptn = parse_torrent_name(file_name)
        ptn_tor = parse_torrent_name(tor.name)
        imdb = ia.search_movie(ptn_tor['title']) if 'season' in ptn.keys() else ia.search_movie(ptn['title'])
        imdb = imdb if imdb else ia.search_movie(ptn_tor['title'])
        imdb = imdb if imdb else ia.search_movie(ptn['title'])
        if not imdb_id:
            imdb_id = 'tt'+imdb[0].movieID if imdb else ''
            imdb_id = f"tt{imdb[0].movieID}:{ptn['season']}:{ptn['episode']}" if 'episode' in ptn.keys() else imdb_id
        else:
            imdb_id = imdb_id.split(":")[0]
            imdb_id = f"{imdb_id}:{ptn['season']}:{ptn['episode']}" if 'episode' in ptn.keys() else imdb_id
        server_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_hash}?download=true"
        
        df = pd.DataFrame([{"imdb_id": imdb_id, 'name' : tor.name, 'file_name' : file_name, "url" : server_url, "size" : vd["size"], "time" : time.time(), "hash" : tor.hash}])
        try:
            df.to_sql(name='hftor', con=postgres_engine, if_exists='append', index=False)
        except:
            pass
    qbt_client.torrents_delete(delete_files=True, torrent_hashes=tor.hash)
