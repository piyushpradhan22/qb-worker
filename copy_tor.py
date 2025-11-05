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
    Ultra-comprehensive parser for torrent/file names.
    Extracts title, season, episode from virtually any naming convention.
    
    Supports 100+ naming patterns including:
    - Standard: S01E02, s01e02, S01.E02, S01-E02, S1E2
    - Multi-episode: S01E01E02, S01E01-E03, S01E01-03
    - Verbose: Season 1 Episode 2, season1episode2
    - International: Temporada 1 Capitulo 2 (Spanish), Saison 1 Episode 2 (French)
    - X format: 1x02, 01x02, 1X02
    - Part/Chapter: Part 1, P1, Chapter 1, Ch.1, Part1Ep2
    - Episode only: Episode 1, Ep1, E01, #01
    - Numbered: 101 (S01E01), 1001 (S10E01)
    - Bracketed: [01], (01), [S01E01]
    - Anime: - 01, 【01】, [Group] Show - 01
    - Special: S00E01, OVA, SP01
    - And many more...
    
    Returns dict with 'title' (always), 'season' (always, defaults to 1), 'episode' (if found)
    """
    result = {}
    original_name = name
    
    # Remove file extension
    name_clean = re.sub(r'\.(mkv|mp4|avi|mov|wmv|flv|webm|m4v|mpg|mpeg)$', '', name, flags=re.IGNORECASE)
    
    # Season and Episode patterns (ordered by specificity to avoid false matches)
    patterns = [
        # 1. MULTI-EPISODE FORMATS (must come first to catch ranges)
        # S01E01E02E03, S01E01-E03, S01E01-03
        (r'[Ss](\d{1,2})[\s\.\-_]*[Ee](\d{1,3})(?:[\s\.\-_]*[Ee]\d{1,3})+', True, 'multi'),
        
        # 2. STANDARD S##E## FORMATS (most common)
        # S01E02, S01 E02, S01.E02, S01-E02, S01_E02, s01e02, S1E2
        (r'[Ss](\d{1,2})[\s\.\-_]*[Ee](\d{1,3})', True, 'standard'),
        
        # 3. BRACKETED STANDARD FORMATS
        # [S01E01], (S01E01), [01x01]
        (r'\[[\s]*[Ss]?(\d{1,2})[\s\.\-_]*[xXeE][\s]*(\d{1,3})[\s]*\]', True, 'bracketed'),
        (r'\([\s]*[Ss]?(\d{1,2})[\s\.\-_]*[xXeE][\s]*(\d{1,3})[\s]*\)', True, 'bracketed'),
        
        # 4. VERBOSE SEASON/EPISODE (English)
        # Season 01 Episode 02, season 1 episode 2, SEASON01EPISODE02
        (r'[Ss]eason[\s\.\-_]*(\d{1,2})[\s\.\-_]*[Ee]pisode[\s\.\-_]*(\d{1,3})', True, 'verbose'),
        
        # 5. INTERNATIONAL VERBOSE FORMATS
        # Spanish: Temporada 1 Capitulo 2, Temporada 1 Episodio 2
        (r'[Tt]emporada[\s\.\-_]*(\d{1,2})[\s\.\-_]*(?:[Cc]apitulo|[Ee]pisodio)[\s\.\-_]*(\d{1,3})', True, 'spanish'),
        # French: Saison 1 Episode 2
        (r'[Ss]aison[\s\.\-_]*(\d{1,2})[\s\.\-_]*[Ee]pisode[\s\.\-_]*(\d{1,3})', True, 'french'),
        # German: Staffel 1 Episode 2
        (r'[Ss]taffel[\s\.\-_]*(\d{1,2})[\s\.\-_]*[Ee]pisode[\s\.\-_]*(\d{1,3})', True, 'german'),
        
        # 6. PART/CHAPTER WITH EPISODE
        # Part 1 Episode 2, P1E2, Part1Ep2, Chapter 1 Episode 2, Ch1E2
        (r'[Pp](?:art|t)?[\s\.\-_]*(\d{1,2})[\s\.\-_]*[Ee](?:p|pisode)?[\s\.\-_]*(\d{1,3})', True, 'part'),
        (r'[Cc](?:hapter|h)?[\s\.\-_]*(\d{1,2})[\s\.\-_]*[Ee](?:p|pisode)?[\s\.\-_]*(\d{1,3})', True, 'chapter'),
        
        # 7. X FORMAT
        # 1x02, 01x02, 1X02, 10x3
        (r'(\d{1,2})[xX](\d{1,3})', True, 'x_format'),
        
        # 8. DASH FORMAT (season-episode)
        # 1-01, 01-01, S1-E1
        (r'[Ss]?(\d{1,2})[\s]*\-[\s]*[Ee]?(\d{1,3})(?:\D|$)', True, 'dash'),
        
        # 9. NUMBERED FORMAT (3-4 digits where first 1-2 = season, last 2 = episode)
        # 101 = S01E01, 1001 = S10E01, 0101 = S01E01
        # BUT NOT years like 2010, 2020 etc (must not be preceded by word boundary)
        (r'(?<![\d\w])(\d{1,2})(\d{2})(?![\d])', True, 'numbered'),
        
        # 10. ANIME SPECIFIC FORMATS
        # [Group] Show - 01, Show - 01 -, 【01】
        (r'[\-\s][\s]*(\d{1,3})[\s]*(?:\-|$)', False, 'anime_dash'),
        (r'【(\d{1,3})】', False, 'anime_jp_bracket'),
        (r'「(\d{1,3})」', False, 'anime_jp_quote'),
        
        # 11. BRACKETED EPISODE ONLY
        # [01], (01), [Episode 01]
        (r'\[[\s]*[Ee]pisode[\s]*(\d{1,3})[\s]*\]', False, 'bracketed_ep'),
        (r'\[[\s]*(\d{1,3})[\s]*\]', False, 'bracketed_num'),
        (r'\([\s]*(\d{1,3})[\s]*\)', False, 'paren_num'),
        
        # 12. EPISODE VERBOSE (multiple languages)
        # Episode 1, episode 01, EPISODE 5
        (r'[Ee]pisode[\s\.\-_]*(\d{1,3})(?:\D|$)', False, 'episode_en'),
        # Episodio (Spanish)
        (r'[Ee]pisodio[\s\.\-_]*(\d{1,3})(?:\D|$)', False, 'episode_es'),
        # Capitulo (Spanish)
        (r'[Cc]apitulo[\s\.\-_]*(\d{1,3})(?:\D|$)', False, 'episode_cap'),
        # Folge (German)
        (r'[Ff]olge[\s\.\-_]*(\d{1,3})(?:\D|$)', False, 'episode_de'),
        
        # 13. EPISODE ABBREVIATED
        # Ep 5, EP.12, ep-03, Ep.1
        (r'[Ee]p[\s\.\-_]*(\d{1,3})(?:\D|$)', False, 'episode_abbr'),
        
        # 14. PART/CHAPTER ONLY (as episode, season = 1)
        # Part 1, P1, Pt1, Pt.1
        (r'[Pp](?:art|t)?[\s\.\-_]*(\d{1,3})(?:\D|$)', False, 'part_only'),
        # Chapter 1, Ch1, Ch.1
        (r'[Cc](?:hapter|h)?[\s\.\-_]*(\d{1,3})(?:\D|$)', False, 'chapter_only'),
        
        # 15. SPECIAL FORMATS
        # #01, No.01, №01
        (r'[#№][\s]*(\d{1,3})(?:\D|$)', False, 'hash'),
        (r'[Nn]o\.[\s]*(\d{1,3})(?:\D|$)', False, 'number'),
        
        # 16. VOLUME/TOME (treating as episode)
        # Vol.1, Volume 1, V1, Tome 1
        (r'[Vv](?:ol|olume)?[\s\.\-_]*(\d{1,3})(?:\D|$)', False, 'volume'),
        (r'[Tt]ome[\s\.\-_]*(\d{1,3})(?:\D|$)', False, 'tome'),
        
        # 17. DISC/DVD FORMAT (treating as episode)
        # Disc 1, DVD1, CD1
        (r'[Dd](?:isc|isk|vd)[\s\.\-_]*(\d{1,3})(?:\D|$)', False, 'disc'),
        (r'[Cc][Dd][\s\.\-_]*(\d{1,3})(?:\D|$)', False, 'cd'),
        
        # 18. MINISERIES FORMATS
        # Night 1, Day 1, Part I, Part II
        (r'[Nn]ight[\s\.\-_]*(\d{1,2})(?:\D|$)', False, 'night'),
        (r'[Dd]ay[\s\.\-_]*(\d{1,2})(?:\D|$)', False, 'day'),
        (r'[Pp]art[\s\.\-_]*(I{1,3}|IV|V|VI{0,3}|IX|X)(?:\D|$)', False, 'roman'),
        
        # 19. OF FORMAT
        # 1 of 10, 01 of 10, (1/10)
        (r'(\d{1,3})[\s]*(?:of|/)[\s]*\d{1,3}', False, 'of_format'),
        
        # 20. SHORT E## FORMAT (must be last to avoid false matches)
        # E12, e05, E1
        (r'[Ee](\d{1,3})(?:\D|$)', False, 'e_short'),
    ]
    
    season = None
    episode = None
    match_pos = -1
    matched_pattern = None
    
    for pattern, has_season, pattern_type in patterns:
        match = re.search(pattern, name_clean, re.IGNORECASE)
        if match:
            match_pos = match.start()
            groups = match.groups()
            matched_pattern = pattern_type
            
            # Special handling for multi-episode (extract first episode only)
            if pattern_type == 'multi':
                season = int(groups[0])
                episode = int(groups[1])
            # Special handling for roman numerals
            elif pattern_type == 'roman':
                roman_to_int = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 
                               'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
                season = 1
                episode = roman_to_int.get(groups[0].upper(), 1)
            # Special handling for numbered format (validate it's reasonable)
            elif pattern_type == 'numbered':
                potential_season = int(groups[0])
                potential_episode = int(groups[1])
                # Only use if season is 0-20 and episode is 1-99 (reasonable ranges)
                # Also skip if it looks like a year (19xx, 20xx)
                if (0 <= potential_season <= 20 and 1 <= potential_episode <= 99 and
                    not (potential_season >= 19 and potential_episode >= 0)):  # Skip years
                    season = potential_season if potential_season > 0 else 1
                    episode = potential_episode
                else:
                    continue  # Skip this match, try next pattern
            # Standard handling
            elif has_season and len(groups) >= 2:
                season = int(groups[0])
                episode = int(groups[1])
            elif not has_season and len(groups) >= 1:
                season = 1  # Default season
                episode = int(groups[0])
            
            if season is not None and episode is not None:
                break
    
    # Extract title (everything before season/episode info or quality tags)
    if match_pos > 0:
        title_part = name_clean[:match_pos]
    else:
        # No season/episode found, try to extract before quality/year info
        quality_match = re.search(r'(720p|1080p|2160p|4k|480p|576p|\d{4}|bluray|brrip|webrip|web-dl|hdtv|sdtv|dvdrip|xvid|x264|x265|h264|h265|hevc|avc|10bit|hdr)', 
                                  name_clean, re.IGNORECASE)
        if quality_match:
            title_part = name_clean[:quality_match.start()]
        else:
            title_part = name_clean
    
    # Clean up the title - remove group tags, brackets, special chars
    # Remove common group tags like [GroupName] or (GroupName) at the start
    title = re.sub(r'^[\[\(][\w\s\-]+[\]\)][\s\.\-_]*', '', title_part)
    # Remove anime-style separators at the end
    title = re.sub(r'[\s\.\-_]*[\-\~]+[\s]*$', '', title)
    # Replace separators with spaces
    title = re.sub(r'[\._\-\[\]\(\)]', ' ', title)
    # Remove multiple spaces
    title = re.sub(r'\s+', ' ', title).strip()
    
    result['title'] = title if title else original_name
    
    # Always include season - default to 1 if not found
    result['season'] = season if season is not None else 1
    
    if episode is not None:
        result['episode'] = episode
    
    # Add metadata about which pattern matched (for debugging)
    if matched_pattern:
        result['_pattern'] = matched_pattern
    
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
