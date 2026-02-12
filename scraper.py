"""RSS ingestion script that appends fresh items into the existing pool.

This version is lightweight (no transformer/sklearn dependencies) so it can
run in constrained environments while still producing all model feature fields.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import json
import random
import re
import socket
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import pandas as pd

try:
    import aiohttp
except Exception:  # pragma: no cover - optional dependency
    aiohttp = None

socket.setdefaulttimeout(10)

TARGET_MIN = 10000
TARGET_MAX = 10000
TEXT_LIMIT = 2048
MERGE_WITH_EXISTING = True
WISHLIST_PATH = Path("data/scraper_wishlist.json")
USE_WISHLIST = False
USE_GOOGLE_NEWS = True
FILTER_GOOGLE_NEWS_FROM_POOL = False
STRICT_SOURCE_FILTER = False
MAX_GOOGLE_QUERIES = 120
MAX_GOOGLE_NEWS_ITEMS = 8500
MAX_DOMAIN_SHARE = 0.12
MAX_PER_DOMAIN = 150
GOOGLE_SOURCE_RE = re.compile(r"<font[^>]*>([^<]+)</font>", flags=re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9]+")
BAD_HEADLINE_RE = re.compile(r"^\s*(live updates?|watch live|breaking:?)\s*$", flags=re.IGNORECASE)
SCRAPER_USER_AGENT = (
    "AddictionEngineResearchBot/1.0 "
    "(respectful scraper; contact: local-dev)"
)
FEED_FETCH_WORKERS = 20
DEFAULT_FETCH_MODE = "thread"
MAX_PARALLEL_REQUESTS = 15
ASYNC_TASK_BATCH_SIZE = 2000
GOOGLE_TIME_SLICE_HOURS = 0
GOOGLE_TIME_SLICE_LOOKBACK_HOURS = 48

GOOGLE_NEWS_LOCALES = [
    ("US", "en-US", "US:en"),
    ("GB", "en-GB", "GB:en"),
    ("CA", "en-CA", "CA:en"),
    ("AU", "en-AU", "AU:en"),
    ("IN", "en-IN", "IN:en"),
    ("IE", "en-IE", "IE:en"),
    ("NZ", "en-NZ", "NZ:en"),
    ("SG", "en-SG", "SG:en"),
    ("PH", "en-PH", "PH:en"),
    ("MY", "en-MY", "MY:en"),
    ("ZA", "en-ZA", "ZA:en"),
    ("NG", "en-NG", "NG:en"),
    ("KE", "en-KE", "KE:en"),
    ("PK", "en-PK", "PK:en"),
]

# Broad niche expansion for diversity-mode scraping.
DIVERSITY_NICHES = [
    "Urban Gardening",
    "Obscure History",
    "Solarpunk",
    "Street Food Trends",
    "Analog Photography",
    "Minimalist Running",
    "Tiny Home Architecture",
    "Retro Handheld Gaming",
    "Vintage Computer Restoration",
    "Public Transit Design",
    "Civic Tech",
    "Neighborhood Mutual Aid",
    "Open Hardware",
    "3D Printing Furniture",
    "DIY Electronics",
    "Maker Education",
    "Low-Waste Living",
    "Circular Fashion",
    "Community Composting",
    "Urban Beekeeping",
    "Indoor Hydroponics",
    "Home Fermentation",
    "Specialty Coffee",
    "Tea Culture",
    "Artisanal Bread",
    "Food Preservation",
    "Culinary Anthropology",
    "Gastropolitics",
    "Digital Nomad Hubs",
    "Remote Work Rituals",
    "Workplace Psychology",
    "Burnout Recovery",
    "Deep Work Methods",
    "Life Hack Experiments",
    "Personal Knowledge Management",
    "Vibe Coding",
    "Prompt Engineering Tactics",
    "No-Code Automation",
    "Open Source AI Tools",
    "AI Safety Governance",
    "Algorithmic Transparency",
    "Data Privacy Rights",
    "Digital Identity",
    "Online Community Governance",
    "Platform Labor",
    "Creator Economy",
    "Niche Subculture Drama",
    "Internet Slang Evolution",
    "Aesthetic Microtrends",
    "Piratecore",
    "Corporate Goth",
    "Gorpcore",
    "Indie Sleaze",
    "Librarian Chic",
    "Digital Minimalism",
    "Silent Walk Movement",
    "Urban Exploration",
    "Abandoned Infrastructure",
    "Railway Ghost Stations",
    "Industrial Ruins",
    "Lidar Archaeology",
    "Underwater Archaeology",
    "Lost Civilizations",
    "Museum Tech",
    "Public History",
    "Science Communication",
    "Citizen Science",
    "Bird Migration Tracking",
    "Marine Biology Discoveries",
    "Mycology Discoveries",
    "Extreme Weather Adaptation",
    "Climate Resilience",
    "Grid Modernization",
    "Energy Storage",
    "Heat Pump Adoption",
    "Walkable Cities",
    "Bike Infrastructure",
    "Micromobility Policy",
    "Affordable Housing Models",
    "Cooperative Housing",
    "Public Health Data",
    "Longevity Research",
    "Biohacking Protocols",
    "Preventive Medicine",
    "Sleep Science",
    "Neurotechnology",
    "Brain-Computer Interface",
    "Quantum Computing",
    "Synthetic Biology",
    "Robotics at Home",
    "Space Policy",
    "Satellite Infrastructure",
    "Ocean Cable Systems",
    "Supply Chain Forensics",
    "Forensic Accounting",
    "Antitrust Case Studies",
    "Regulatory Investigations",
    "Labor Strikes",
    "Whistleblower Reports",
    "True Crime Investigations",
    "Courtroom Technology",
]

DIVERSITY_QUERY_TEMPLATES = (
    "{niche} 2026 trends",
    "{niche} deep dive",
    "{niche} latest news",
    "{niche} case studies",
)

ALLOWED_SOURCE_SUFFIXES = (
    "dailymail.co.uk",
    "thedailybeast.com",
    "breitbart.com",
    "nationalreview.com",
    "tmz.com",
    "zerohedge.com",
    "theintercept.com",
    "theguardian.com",
    "propublica.org",
    "thebulletin.org",
    "commondreams.org",
    "cointelegraph.com",
    "businessinsider.com",
    "forbes.com",
    "hypebeast.com",
    "rollingstone.com",
    "vulture.com",
    "dazeddigital.com",
    "ecowatch.com",
    "grist.org",
    "inhabitat.com",
    "statnews.com",
    "healthline.com",
    "fiercebiotech.com",
    "themarshallproject.org",
    "legalevolution.org",
    "outsideonline.com",
    "apartmenttherapy.com",
)


def get_creative_style(emotion_label: str) -> str:
    style_map = {
        "anger": "Angry Rant",
        "fear": "Dystopian Warning",
        "joy": "Hype Beast",
        "sadness": "Somber Reflection",
        "surprise": "Clickbait Shocker",
        "disgust": "Sarcastic Takedown",
        "neutral": "Analytical Deep Dive",
    }
    return style_map.get(emotion_label, "Standard Reporting")


def load_scraper_wishlist(path: Path = WISHLIST_PATH) -> dict[str, list[str]]:
    if not path.exists():
        return {"queries": [], "preferred_domains": [], "extra_feed_urls": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"queries": [], "preferred_domains": [], "extra_feed_urls": []}

    out: dict[str, list[str]] = {}
    for key in ("queries", "preferred_domains", "extra_feed_urls"):
        value = payload.get(key, [])
        if not isinstance(value, list):
            out[key] = []
            continue
        clean = [str(x).strip() for x in value if str(x).strip()]
        out[key] = list(dict.fromkeys(clean))
    return out


def is_allowed_source_domain(domain: str) -> bool:
    d = str(domain).strip().lower()
    if not d:
        return False
    return any(d == suffix or d.endswith(f".{suffix}") for suffix in ALLOWED_SOURCE_SUFFIXES)


def build_diversity_queries() -> list[str]:
    queries: list[str] = []
    for niche in DIVERSITY_NICHES[:100]:
        for template in DIVERSITY_QUERY_TEMPLATES:
            queries.append(template.format(niche=niche))
    return list(dict.fromkeys(queries))


def build_query_variants(queries: list[str], *, diversity_mode: bool) -> list[str]:
    if not diversity_mode:
        return list(dict.fromkeys(queries))
    expanded: list[str] = []
    for q in queries:
        expanded.extend(
            [
                q,
                f"{q} when:7d",
                f"{q} when:30d",
            ]
        )
    return list(dict.fromkeys(expanded))


def build_time_sliced_queries(
    queries: list[str],
    *,
    slice_hours: int,
    lookback_hours: int,
    now_utc: datetime | None = None,
) -> list[str]:
    if slice_hours <= 0:
        return list(dict.fromkeys(queries))
    lookback_hours = max(slice_hours, int(lookback_hours))
    now = now_utc or datetime.now(timezone.utc)
    start = now - timedelta(hours=lookback_hours)
    expanded: list[str] = []
    t0 = start
    while t0 < now:
        t1 = min(t0 + timedelta(hours=slice_hours), now)
        after_str = t0.strftime("%Y-%m-%dT%H:%M:%S")
        before_str = t1.strftime("%Y-%m-%dT%H:%M:%S")
        for q in queries:
            expanded.append(f"{q} after:{after_str} before:{before_str}")
        t0 = t1
    return list(dict.fromkeys(expanded))


def build_feed_list(
    wishlist: dict[str, list[str]] | None = None,
    *,
    use_google_news: bool = USE_GOOGLE_NEWS,
    max_google_queries: int = MAX_GOOGLE_QUERIES,
    diversity_mode: bool = False,
    slice_hours: int = GOOGLE_TIME_SLICE_HOURS,
    slice_lookback_hours: int = GOOGLE_TIME_SLICE_LOOKBACK_HOURS,
) -> list[str]:
    if wishlist is None:
        wishlist = {"queries": [], "preferred_domains": [], "extra_feed_urls": []}
    manifest_feeds = [
        # Manifest-only source list.
        # Policy / engagement
        "https://www.dailymail.co.uk/articles.rss",
        "https://www.thedailybeast.com/rss",
        "https://www.breitbart.com/feed/",
        "https://www.nationalreview.com/feed/",
        "https://www.tmz.com/rss.xml",

        # Systemic / resilience reporting
        "http://feeds.feedburner.com/zerohedge",
        "https://theintercept.com/feed/?rss",
        "https://www.theguardian.com/news/series/the-long-read/rss",
        "https://www.propublica.org/feeds/propublica/main",
        "https://thebulletin.org/feed/",
        "https://www.commondreams.org/rss.xml",

        # Growth / achievement / aspiration
        "https://cointelegraph.com/rss",
        "https://www.businessinsider.com/rss",
        "https://www.forbes.com/innovation/feed/",
        "https://hypebeast.com/feed",
    ]
    # Broader sources for the same topic families to reduce repetition.
    expanded_feeds = [
        # Tech / policy / business
        "https://www.theverge.com/rss/index.xml",
        "https://techcrunch.com/feed/",
        "https://www.wired.com/feed/rss",
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://www.engadget.com/rss.xml",
        "https://www.zdnet.com/news/rss.xml",
        "https://www.cnet.com/rss/news/",
        "https://www.technologyreview.com/feed/",
        "https://venturebeat.com/category/ai/feed/",
        "https://www.androidauthority.com/feed/",
        "https://www.macrumors.com/macrumors.xml",
        "https://9to5mac.com/feed/",
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "http://rss.cnn.com/rss/cnn_business.rss",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        # Gaming / culture
        "https://www.ign.com/rss",
        "https://www.polygon.com/rss/index.xml",
        "https://www.eurogamer.net/feed",
        "https://www.pcgamer.com/rss/",
        "https://www.destructoid.com/feed/",
        "https://www.rockpapershotgun.com/feed",
        "https://www.gamespot.com/feeds/mashup/",
        "https://kotaku.com/rss",
        # Long-read / systems / civic
        "https://www.goodnewsnetwork.org/feed/",
        "https://www.propublica.org/feeds/propublica/main",
        "https://www.theguardian.com/world/rss",
        "https://www.theguardian.com/environment/rss",
        # Curiosity / utility
        "https://lifehacker.com/rss",
        "https://www.atlasobscura.com/feeds/latest",
        "https://www.sciencealert.com/feed",
        # Culture / micro-trends / internet subcultures
        "https://www.rollingstone.com/feed/",
        "https://www.vulture.com/rss/index.xml",
        "https://www.dazeddigital.com/rss",
        # Sustainability / circular systems
        "https://www.ecowatch.com/feed",
        "https://grist.org/feed/",
        "https://inhabitat.com/feed/",
        # Health tech / frontier biology
        "https://www.statnews.com/feed/",
        "https://www.healthline.com/health-news/feed",
        "https://www.fiercebiotech.com/rss/xml",
        # Legal / policy thrillers / investigations
        "https://www.themarshallproject.org/feed",
        "https://www.legalevolution.org/feed/",
        # Lifestyle / DIY / remote work / home optimization
        "https://www.outsideonline.com/feed/",
        "https://www.apartmenttherapy.com/main.rss",
    ]
    base_feeds = list(dict.fromkeys(manifest_feeds + expanded_feeds))
    queries = [
        # PILLAR 1: Structural Policy & Geopolitics (Hard News)
        "Interstate trade metrics",
        "Ukraine winter offensive logistics",
        "Energy grid stability",
        "Strategic autonomy EU",
        "Labor union density 2026",
        "NATO red line escalation",
        "China tariff retaliation",
        "Constitutional crisis legal analysis",
        "Border security tech",
        # PILLAR 2: Macro Resilience & Economy (Anxiety/Action)
        "Hyper-inflation 2026 index",
        "Corporate debt cycles",
        "Venture capital winter 2026",
        "Wage growth automation statistics",
        "Retail spending shifts",
        "Bank failure risk assessment",
        "Circular economy startups",
        "Microplastics cognitive decline study",
        "Food security breakthroughs",
        # PILLAR 3: Frontier Progress & Aspiration (Hype/FOMO)
        "Synthetic biology standards",
        "Brain-computer interface breakthrough",
        "Desktop AI agents launch",
        "Satellite-to-cell infrastructure",
        "Quantum computing commercialization",
        "Biohacking 2026 protocols",
        "Longevity supplements latest research",
        "Nvidia Blackwell power wire report",
        "Sora AI video launch",
        "Luxury mansion Miami market",
        "First-mover crypto assets",
        "Billionaire productivity routines",
        # PILLAR 4: Internet Culture & Micro-Drama (Social/Identity)
        "Piratecore aesthetic trend",
        "Librarian Chic fashion",
        "Indie Sleaze revival 2026",
        "Corporate Dropout lifestyle",
        "De-influencing viral series",
        "Deepfake celebrity scandal",
        "Creator economy exposed",
        "TikTok Shop live trends",
        "Gen Alpha content norms",
        "Viral social media challenges injuries",
        "Super Bowl LX ad influence index",
        # PILLAR 5: Daily Routine & Satisfying Utility (Filler/Retention)
        "One-minute wealth hacks",
        "AI-assisted DIY home improvement",
        "Productivity Vibe Coding",
        "Cozy gaming industry updates",
        "Satisfying restoration process",
        "Gastropolitics food trends",
        "Digital nomad hubs 2026",
        "Best running shoes 2026 reviews",
        "Time-saving automation hacks",
        # PILLAR 6: Science & Global Systems (Wonder/Morbid Curiosity)
        "Deep sea fungi discovery",
        "Bird flu human transmission risk",
        "Topological qubits stability",
        "Archaeology breakthrough Lidar",
        "True crime policy thriller",
        "Unsolved systemic poverty report",
        "Space launch failure analysis",
        "Forever chemicals blood test results",
        # Social Survival: trigger "Us vs. Them"
        "Workplace toxicity 2026 leaks",
        "De-influencing viral brands",
        "Cancel culture case studies",
        "Corporate social credit score",
        # Hidden Knowledge: trigger the "Secret Edge"
        "Proprietary AI prompting secrets",
        "Hidden banking fee loopholes",
        "Black market tech repair",
        "Non-standard biohacking results",
        # Micro-Drama: trigger "Moral Arbitrage"
        "Niche subculture drama February 2026",
        "Hobby community gatekeeping",
        "Influencer legal battles",
        "Gaming industry whistleblower",
        # Aesthetic Identity: trigger "Belonging"
        "Solarpunk lifestyle 2026",
        "Corporate Goth fashion",
        "Silent walk movement",
        "Digital minimalism hardware",
        # The Hustle Loop: trigger "FOMO"
        "Passive income automation 2026",
        "Early-access waitlist leaks",
        "Venture capital ghost jobs",
        "Startup equity traps",
        # Systemic Mystery: trigger "Morbid Curiosity"
        "Unresolved satellite interference",
        "Ocean bed data cable mystery",
        "Forgotten infrastructure ruins",
        "Deep sea archeology Lidar",
        "Lidar archeology discoveries 2026",
        "Abandoned infrastructure ruins",
        "Solarpunk city design",
        "Piratecore aesthetic history",
        "Lost underwater civilizations lidar",
    ]
    prioritized_queries: list[str] = []
    if wishlist["queries"]:
        prioritized_queries.extend(wishlist["queries"])
    # Prefer direct feed URLs over site: query wrappers to avoid heavy news.google.com
    # concentration in the final pool.
    if wishlist["extra_feed_urls"]:
        base_feeds.extend(wishlist["extra_feed_urls"])

    diversity_queries = build_diversity_queries() if diversity_mode else []
    query_pool = list(dict.fromkeys(prioritized_queries + queries + diversity_queries))
    query_pool = build_query_variants(query_pool, diversity_mode=diversity_mode)
    query_pool = build_time_sliced_queries(
        query_pool,
        slice_hours=slice_hours,
        lookback_hours=slice_lookback_hours,
    )
    query_pool = query_pool[:max_google_queries]

    google_news_query_feeds = []
    if use_google_news:
        for query in query_pool:
            for gl, hl, ceid in GOOGLE_NEWS_LOCALES:
                google_news_query_feeds.append(
                    f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}"
                )

    # Stable de-dupe of feed URLs.
    # Query-driven feeds come first so niche/sliced discovery claims keys before mainstream overlap.
    return list(dict.fromkeys(google_news_query_feeds + base_feeds))


def extract_domain(link: str) -> str:
    clean = str(link).replace("https://", "").replace("http://", "")
    return clean.split("/")[0].replace("www.", "").strip().lower() or "unknown"


def is_high_quality_item(headline: str, summary: str, link: str) -> bool:
    h = str(headline or "").strip()
    s = str(summary or "").strip()
    l = str(link or "").strip()
    if not h or not l:
        return False
    if len(h) < 25:
        return False
    if len(WORD_RE.findall(h)) < 5:
        return False
    if BAD_HEADLINE_RE.match(h):
        return False
    if "photo gallery" in h.lower() or "podcast" in h.lower():
        return False
    # For direct RSS sources, require a little context beyond just the title.
    if len(s) < 40 and extract_domain(l) != "news.google.com":
        return False
    return True


def source_label_to_topic(source_label: str) -> str:
    label = str(source_label or "").strip().lower()
    if not label:
        return ""
    # If it already looks like a domain, keep it as a domain.
    if "." in label and " " not in label:
        return label.replace("www.", "")
    # Keep a stable source namespace for non-domain labels.
    slug = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    return f"source_{slug}" if slug else ""


def extract_google_source_label(summary: str, headline: str) -> str:
    m = GOOGLE_SOURCE_RE.findall(str(summary or ""))
    if m:
        return str(m[-1]).strip()
    head = str(headline or "")
    if " - " in head:
        return head.rsplit(" - ", 1)[-1].strip()
    return ""


def topic_name_from_row(link: str, headline: str, summary: str) -> str:
    raw_domain = extract_domain(link)
    if raw_domain != "news.google.com":
        return raw_domain
    source = extract_google_source_label(summary, headline)
    topic = source_label_to_topic(source)
    return topic or raw_domain


def cap_domain_mix(df: pd.DataFrame, target_max: int, max_google_news_items: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Topic_Name"] = out.apply(
        lambda r: topic_name_from_row(
            str(r.get("Link", "")),
            str(r.get("Headline", "")),
            str(r.get("Summary", "")),
        ),
        axis=1,
    )
    out["Raw_Domain"] = out["Link"].apply(extract_domain)
    out["Published_TS"] = pd.to_datetime(out.get("Published", ""), errors="coerce", utc=True)
    out["Published_TS"] = out["Published_TS"].fillna(pd.Timestamp.now(tz="UTC"))
    out = out.sort_values("Published_TS", ascending=False).reset_index(drop=True)

    # Global per-domain cap and stricter cap for Google News wrappers.
    generic_cap = max(1, min(MAX_PER_DOMAIN, int(target_max * MAX_DOMAIN_SHARE)))
    domain_cap = {d: generic_cap for d in out["Topic_Name"].unique()}
    raw_google_cap = min(max_google_news_items, target_max)

    picks = []
    domain_counts: dict[str, int] = {}
    raw_google_count = 0
    for row in out.itertuples(index=False):
        domain = getattr(row, "Topic_Name")
        raw_domain = getattr(row, "Raw_Domain")
        if raw_domain == "news.google.com" and raw_google_count >= raw_google_cap:
            continue
        if domain_counts.get(domain, 0) >= domain_cap.get(domain, generic_cap):
            continue
        picks.append(row)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if raw_domain == "news.google.com":
            raw_google_count += 1
        if len(picks) >= target_max:
            break

    limited = pd.DataFrame(picks, columns=out.columns)
    return limited.drop(
        columns=["Topic_Name", "Raw_Domain", "Published_TS"], errors="ignore"
    ).reset_index(drop=True)


def _combined_text_tokens(headline: str, summary: str = "", topic_name: str = "") -> tuple[str, set[str]]:
    text = " ".join(
        [
            str(headline or ""),
            str(summary or ""),
            str(topic_name or ""),
        ]
    ).lower()
    tokens = set(WORD_RE.findall(text))
    return text, tokens


def _matches(text: str, tokens: set[str], phrase: str) -> bool:
    p = str(phrase or "").strip().lower()
    if not p:
        return False
    if " " in p:
        return p in text
    return p in tokens


def entity_key(headline: str, summary: str = "", topic_name: str = "") -> str:
    text, tokens = _combined_text_tokens(headline, summary, topic_name)

    mapping = [
        ("geopolitics_conflict", [
            "ukraine", "russia", "nato", "china", "taiwan", "border", "missile", "gaza", "israel",
            "iran", "north korea", "ceasefire", "offensive", "defense ministry",
        ]),
        ("structural_governance", [
            "starmer", "labour", "tory", "trump", "senate", "congress", "ftc", "supreme court",
            "election", "policy", "regulator", "antitrust", "constitutional",
        ]),
        ("macro_economics", [
            "inflation", "tariff", "recession", "interest rate", "fed", "ecb", "debt", "gdp",
            "labor market", "jobs report", "bond yields",
        ]),
        ("systemic_health", [
            "alzheimer", "cancer", "cholera", "bird flu", "antibiotic resistance", "forever chemicals",
            "microplastic", "public health", "epidemic",
        ]),
        ("frontier_science", [
            "quantum", "synthetic biology", "biotech", "fusion", "satellite", "lidar",
            "space telescope", "astronomy", "nasa", "deep sea",
        ]),
        ("sustainability_climate", [
            "climate", "sustainability", "renewable", "solar", "wind", "ocean plastic",
            "grid modernization", "heat pump", "circular economy",
        ]),
        ("culture_identity", [
            "piratecore", "solarpunk", "gorpcore", "indie sleaze", "librarian chic",
            "subculture", "aesthetic", "trend", "viral",
        ]),
        ("lifestyle_hacks", [
            "lifehack", "diy", "productivity", "vibe coding", "digital nomad", "street food",
            "apartment therapy", "biohacking",
        ]),
        ("gaming_entertainment", [
            "gaming", "game", "xbox", "playstation", "nintendo", "steam", "pcgamer", "kotaku",
            "eurogamer", "destructoid", "polygon", "gamespot", "trailer",
        ]),
        ("true_crime_investigation", [
            "cold case", "homicide", "investigation", "whistleblower", "crime", "court filing",
            "propublica", "marshall project",
        ]),
        ("finance_markets", [
            "sp500", "nasdaq", "ipo", "bitcoin", "crypto", "market", "earnings", "revenue",
            "profit", "venture capital",
        ]),
        # Keep major tech companies as distinct entities for personalization.
        ("openai", ["openai", "chatgpt", "sam altman", "sora", "gpt-5", "o1-preview"]),
        ("google", ["google", "alphabet", "youtube", "gemini", "deepmind", "waymo", "pixel"]),
        ("apple", ["apple", "iphone", "macbook", "vision pro", "ios", "tim cook"]),
        ("meta", ["meta", "facebook", "instagram", "threads", "llama", "quest"]),
        ("microsoft", ["microsoft", "windows", "azure", "copilot", "nadella"]),
        ("amazon", ["amazon", "aws", "blue origin", "bezos", "kindle"]),
        ("tesla_musk", ["tesla", "spacex", "elon musk", "xai", "grok", "starship", "neuralink"]),
        ("nvidia_chips", ["nvidia", "amd", "tsmc", "intel", "h100", "blackwell", "semiconductor"]),
    ]
    for key, keywords in mapping:
        if any(_matches(text, tokens, k) for k in keywords):
            return key
    return "unknown_entity"


def event_key(headline: str, summary: str = "", topic_name: str = "") -> str:
    text, tokens = _combined_text_tokens(headline, summary, topic_name)
    mapping = {
        "ai_breakthrough": [
            "artificial intelligence", "llm", "agentic", "inference", "model release", "chatgpt",
            "frontier model",
        ],
        "science_space": [
            "nasa", "astronomy", "biotech", "dna", "quantum", "fusion", "space launch",
            "lidar", "synthetic biology",
        ],
        "business_deal": [
            "acquires", "acquisition", "merger", "buyout", "investment", "series", "funding",
            "venture capital", "raises",
        ],
        "financial_report": [
            "quarterly", "earnings", "revenue", "profit", "guidance", "fiscal", "q1", "q2", "q3", "q4",
        ],
        "legal_clash": [
            "antitrust", "lawsuit", "suing", "regulation", "eu fine", "court ruling", "injunction",
            "ftc", "doj",
        ],
        "cyber_threat": [
            "hacked", "data breach", "ransomware", "exploit", "zero-day", "cybersecurity",
            "vulnerability", "cve-",
        ],
        "product_drop": [
            "unveils", "launch", "launched", "announced", "pre-order", "leak", "rumor", "hands-on",
        ],
        "macro_shift": [
            "layoffs", "job cuts", "hiring freeze", "recession", "economic boom", "inflation",
            "interest rate",
        ],
        "policy_shift": [
            "bill passed", "policy change", "executive order", "regulatory framework", "governance",
            "labor law",
        ],
        "social_backlash": [
            "backlash", "outrage", "boycott", "cancel culture", "viral criticism", "scandal",
        ],
        "crime_probe": [
            "investigation", "charged", "indicted", "cold case", "whistleblower", "trial",
        ],
        "culture_trend": [
            "trend", "aesthetic", "subculture", "viral", "micro-trend", "fashion core",
        ],
        "lifestyle_guide": [
            "how to", "guide", "tips", "review", "best", "deal",
        ],
    }
    for key, keywords in mapping.items():
        if any(_matches(text, tokens, k) for k in keywords):
            return key
    return "unknown_event"


def recency_bucket(hours_old: float) -> int:
    if hours_old <= 1:
        return 0
    if hours_old <= 6:
        return 1
    if hours_old <= 24:
        return 2
    if hours_old <= 72:
        return 3
    return 4


def assign_topic_clusters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["Topic_Cluster_ID"] = []
        return df

    n_clusters = max(8, min(64, max(1, len(df) // 20)))
    df["Topic_Cluster_ID"] = (
        df["Headline"].fillna("").astype(str).apply(lambda s: abs(hash(s.lower())) % n_clusters)
    )
    return df


def infer_emotion_simple(text: str) -> str:
    t = str(text).lower()
    rules = {
        "anger": ["outrage", "fury", "backlash", "furious", "rage", "slams", "angry"],
        "fear": ["warn", "threat", "danger", "risk", "crisis", "panic", "fear"],
        "joy": ["wins", "record", "celebrates", "breakthrough", "boost", "surges"],
        "sadness": ["dies", "death", "grief", "mourning", "hospitalized", "decline"],
        "surprise": ["shocker", "unexpected", "sudden", "revealed", "stuns"],
        "disgust": ["scam", "fraud", "corrupt", "toxic", "gross", "disgust"],
    }
    best = "neutral"
    best_score = 0
    for label, words in rules.items():
        score = sum(1 for w in words if w in t)
        if score > best_score:
            best = label
            best_score = score
    return best


def normalize_existing(existing: pd.DataFrame) -> pd.DataFrame:
    out = existing.copy()

    rename = {}
    if "Headline" not in out.columns and "title" in out.columns:
        rename["title"] = "Headline"
    if "Summary" not in out.columns and "description" in out.columns:
        rename["description"] = "Summary"
    if "Link" not in out.columns and "link" in out.columns:
        rename["link"] = "Link"
    if rename:
        out = out.rename(columns=rename)

    if "Headline" not in out.columns:
        out["Headline"] = ""
    if "Summary" not in out.columns:
        out["Summary"] = ""
    if "Link" not in out.columns:
        out["Link"] = ""
    if "Emotion" not in out.columns:
        out["Emotion"] = out["Headline"].apply(infer_emotion_simple)
    if "Vibe_Style" not in out.columns:
        out["Vibe_Style"] = out["Emotion"].apply(get_creative_style)
    if "Published" not in out.columns:
        if "Published_UTC" in out.columns:
            out["Published"] = out["Published_UTC"].astype(str)
        else:
            out["Published"] = ""

    return out[["Headline", "Summary", "Emotion", "Vibe_Style", "Link", "Published"]]


def fetch_feed_entries(feed_url: str) -> tuple[str, list, bool]:
    try:
        parsed_feed = feedparser.parse(feed_url, agent=SCRAPER_USER_AGENT)
        return feed_url, list(getattr(parsed_feed, "entries", [])), bool(getattr(parsed_feed, "bozo", False))
    except Exception:
        return feed_url, [], True


def fetch_feeds_threaded(
    rss_feeds: list[str],
    *,
    max_workers: int,
) -> list[tuple[str, list, bool] | None]:
    parsed_batches: list[tuple[str, list, bool] | None] = [None] * len(rss_feeds)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        total_feeds = len(rss_feeds)
        submit_pulse_every = 10 if total_feeds <= 2000 else 500
        fetch_pulse_every = 50 if total_feeds <= 2000 else 1000
        fetch_start = datetime.now(timezone.utc)
        fetched_count = 0
        for i, feed_url in enumerate(rss_feeds):
            if i % submit_pulse_every == 0:
                print(f"📡 Processing feed {i}/{len(rss_feeds)}: {feed_url[:50]}...")
            futures[executor.submit(fetch_feed_entries, feed_url)] = i

        for future in as_completed(futures):
            idx = futures[future]
            fetched_count += 1
            try:
                parsed_batches[idx] = future.result()
            except Exception:
                parsed_batches[idx] = (rss_feeds[idx], [], True)
            if fetched_count % fetch_pulse_every == 0 or fetched_count == total_feeds:
                elapsed_s = (datetime.now(timezone.utc) - fetch_start).total_seconds()
                rate = fetched_count / max(elapsed_s, 1e-6)
                remaining = total_feeds - fetched_count
                eta_s = remaining / max(rate, 1e-6)
                print(
                    f"✅ Fetched {fetched_count}/{total_feeds} feeds "
                    f"({(fetched_count/total_feeds)*100:.1f}%) "
                    f"| elapsed={elapsed_s:.0f}s eta={eta_s:.0f}s"
                )
    return parsed_batches


async def fetch_feed_entries_async(
    session: "aiohttp.ClientSession",
    feed_url: str,
    semaphore: "asyncio.Semaphore",
) -> tuple[str, list, bool]:
    async with semaphore:
        try:
            async with session.get(feed_url) as response:
                if response.status != 200:
                    return feed_url, [], True
                payload = await response.text(errors="ignore")
        except Exception:
            return feed_url, [], True
    try:
        parsed_feed = feedparser.parse(payload)
        return feed_url, list(getattr(parsed_feed, "entries", [])), bool(getattr(parsed_feed, "bozo", False))
    except Exception:
        return feed_url, [], True


async def fetch_feeds_async_runner(
    rss_feeds: list[str],
    *,
    max_parallel_requests: int,
) -> list[tuple[str, list, bool] | None]:
    if aiohttp is None:
        raise RuntimeError(
            "Async mode requires aiohttp. Install it with: ./venv/bin/pip install aiohttp"
        )

    parsed_batches: list[tuple[str, list, bool] | None] = [None] * len(rss_feeds)
    total_feeds = len(rss_feeds)
    submit_pulse_every = 10 if total_feeds <= 2000 else 500
    fetch_pulse_every = 50 if total_feeds <= 2000 else 1000
    fetch_start = datetime.now(timezone.utc)
    fetched_count = 0

    timeout = aiohttp.ClientTimeout(total=10)
    connector = aiohttp.TCPConnector(limit=max_parallel_requests)
    semaphore = asyncio.Semaphore(max_parallel_requests)
    headers = {"User-Agent": SCRAPER_USER_AGENT}

    async def fetch_with_index(idx: int) -> tuple[int, tuple[str, list, bool]]:
        return idx, await fetch_feed_entries_async(session, rss_feeds[idx], semaphore)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers) as session:
        for batch_start in range(0, total_feeds, ASYNC_TASK_BATCH_SIZE):
            batch_end = min(batch_start + ASYNC_TASK_BATCH_SIZE, total_feeds)
            tasks: list[asyncio.Task] = []
            for i in range(batch_start, batch_end):
                if i % submit_pulse_every == 0:
                    print(f"📡 Processing feed {i}/{total_feeds}: {rss_feeds[i][:50]}...")
                tasks.append(asyncio.create_task(fetch_with_index(i)))

            for task in asyncio.as_completed(tasks):
                fetched_count += 1
                try:
                    idx, batch = await task
                    parsed_batches[idx] = batch
                except Exception:
                    pass
                if fetched_count % fetch_pulse_every == 0 or fetched_count == total_feeds:
                    elapsed_s = (datetime.now(timezone.utc) - fetch_start).total_seconds()
                    rate = fetched_count / max(elapsed_s, 1e-6)
                    remaining = total_feeds - fetched_count
                    eta_s = remaining / max(rate, 1e-6)
                    print(
                        f"✅ Fetched {fetched_count}/{total_feeds} feeds "
                        f"({(fetched_count/total_feeds)*100:.1f}%) "
                        f"| elapsed={elapsed_s:.0f}s eta={eta_s:.0f}s"
                    )
    return parsed_batches


def main(
    mode: str = "standard",
    target_min: int = TARGET_MIN,
    target_max: int = TARGET_MAX,
    use_google_news: bool = USE_GOOGLE_NEWS,
    max_google_queries: int = MAX_GOOGLE_QUERIES,
    max_google_news_items: int = MAX_GOOGLE_NEWS_ITEMS,
    slice_hours: int = GOOGLE_TIME_SLICE_HOURS,
    slice_lookback_hours: int = GOOGLE_TIME_SLICE_LOOKBACK_HOURS,
    fetch_mode: str = DEFAULT_FETCH_MODE,
    max_parallel_requests: int = MAX_PARALLEL_REQUESTS,
) -> None:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "scraped_articles.csv"

    mode = str(mode).strip().lower()
    diversity_mode = mode == "diversity"

    # Auto-scale defaults for large diversity backfill.
    if diversity_mode:
        if target_max == TARGET_MAX:
            target_max = 50000
        if target_min == TARGET_MIN:
            target_min = 25000
        if max_google_queries == MAX_GOOGLE_QUERIES:
            max_google_queries = 1200
        if max_google_news_items == MAX_GOOGLE_NEWS_ITEMS:
            max_google_news_items = 45000
        use_google_news = True

    wishlist = (
        load_scraper_wishlist()
        if USE_WISHLIST
        else {"queries": [], "preferred_domains": [], "extra_feed_urls": []}
    )
    rss_feeds = build_feed_list(
        wishlist=wishlist,
        use_google_news=use_google_news,
        max_google_queries=max_google_queries,
        diversity_mode=diversity_mode,
        slice_hours=slice_hours,
        slice_lookback_hours=slice_lookback_hours,
    )
    print(
        f"Scrape mode={mode} feeds={len(rss_feeds)} "
        f"use_google_news={use_google_news} max_google_queries={max_google_queries} "
        f"max_google_news_items={max_google_news_items} "
        f"slice_hours={slice_hours} lookback_hours={slice_lookback_hours} "
        f"fetch_mode={fetch_mode} max_parallel_requests={max_parallel_requests}"
    )
    if USE_WISHLIST and (wishlist["queries"] or wishlist["preferred_domains"] or wishlist["extra_feed_urls"]):
        print(
            "Wishlist loaded | "
            f"queries={len(wishlist['queries'])} "
            f"preferred_domains={len(wishlist['preferred_domains'])} "
            f"extra_feeds={len(wishlist['extra_feed_urls'])}"
        )
    candidates: list[dict[str, str]] = []
    seen_keys = set()
    google_news_items = 0

    fetch_mode = str(fetch_mode).strip().lower()
    if fetch_mode == "async":
        parsed_batches = asyncio.run(
            fetch_feeds_async_runner(
                rss_feeds,
                max_parallel_requests=max_parallel_requests,
            )
        )
    else:
        parsed_batches = fetch_feeds_threaded(
            rss_feeds,
            max_workers=max_parallel_requests,
        )

    process_pulse_every = 50 if len(parsed_batches) <= 2000 else 1000
    for i, parsed_batch in enumerate(parsed_batches):
        if len(candidates) >= target_max:
            break
        if parsed_batch is None:
            continue
        if i % process_pulse_every == 0:
            print(
                f"🧪 Parsed {i}/{len(parsed_batches)} feeds into candidates "
                f"| candidates={len(candidates)}"
            )
        _, entries, bozo = parsed_batch
        if bozo:
            # Skip malformed feeds gracefully and continue harvesting others.
            pass
        for entry in entries:
            if len(candidates) >= target_max:
                break
            headline = str(getattr(entry, "title", "") or "")
            summary = str(getattr(entry, "summary", "") or "")
            link = str(getattr(entry, "link", "") or "")
            published = (
                str(getattr(entry, "published", "") or "")
                or str(getattr(entry, "updated", "") or "")
                or str(getattr(entry, "pubDate", "") or "")
            )
            key = (link.strip().lower(), headline.strip().lower())
            domain = extract_domain(link)
            if domain == "news.google.com":
                if google_news_items >= max_google_news_items:
                    continue
                google_news_items += 1
                # De-dup Google wrappers by source/topic + headline, not URL token.
                source_topic = topic_name_from_row(link, headline, summary)
                key = (source_topic, headline.strip().lower())
            if key in seen_keys:
                continue

            combined_text = f"{headline} {summary}"[:TEXT_LIMIT]
            if not combined_text.strip():
                continue
            if not is_high_quality_item(headline, summary, link):
                continue
            seen_keys.add(key)
            candidates.append(
                {
                    "Headline": headline,
                    "Summary": summary,
                    "Link": link,
                    "Published": published,
                    "Emotion": infer_emotion_simple(combined_text),
                }
            )

    if candidates:
        new_df = pd.DataFrame(candidates)
        new_df["Vibe_Style"] = new_df["Emotion"].apply(get_creative_style)
        new_df = new_df[["Headline", "Summary", "Emotion", "Vibe_Style", "Link", "Published"]]
        new_df = new_df.drop_duplicates(subset=["Link"]).reset_index(drop=True)
    else:
        if not data_path.exists():
            raise RuntimeError("No records scraped from feeds and no existing pool found.")
        print("No new feed records fetched. Rebuilding feature fields from existing pool.")
        new_df = pd.DataFrame(
            columns=["Headline", "Summary", "Emotion", "Vibe_Style", "Link", "Published"]
        )

    if MERGE_WITH_EXISTING and data_path.exists():
        existing_df = pd.read_csv(data_path)
        existing_df = normalize_existing(existing_df)
        merged = pd.concat([new_df, existing_df], ignore_index=True)
    else:
        merged = new_df

    merged = merged.drop_duplicates(subset=["Link"]).reset_index(drop=True)
    if not merged.empty:
        # Remove near-duplicates that differ only by wrapper URLs.
        merged["_norm_headline"] = (
            merged["Headline"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        if diversity_mode:
            # In diversity mode keep the same headline if it comes from a different source/topic.
            merged["_norm_topic"] = merged.apply(
                lambda r: topic_name_from_row(
                    str(r.get("Link", "")),
                    str(r.get("Headline", "")),
                    str(r.get("Summary", "")),
                ),
                axis=1,
            )
            merged = merged.drop_duplicates(subset=["_norm_headline", "_norm_topic"]).drop(
                columns=["_norm_headline", "_norm_topic"]
            )
        else:
            merged = merged.drop_duplicates(subset=["_norm_headline"]).drop(columns=["_norm_headline"])
        merged = merged.reset_index(drop=True)
    if STRICT_SOURCE_FILTER:
        merged = merged[
            merged["Link"].map(lambda x: is_allowed_source_domain(extract_domain(x)))
        ].reset_index(drop=True)
    if FILTER_GOOGLE_NEWS_FROM_POOL and not merged.empty:
        merged = merged[merged["Link"].map(extract_domain) != "news.google.com"].reset_index(drop=True)
    if len(merged) > target_max:
        merged = cap_domain_mix(merged, target_max, max_google_news_items)

    # Build richer IDs/features.
    df = merged.copy()
    df["Topic_Name"] = df.apply(
        lambda r: topic_name_from_row(
            str(r.get("Link", "")),
            str(r.get("Headline", "")),
            str(r.get("Summary", "")),
        ),
        axis=1,
    )
    df["Topic_ID"], topic_uniques = pd.factorize(df["Topic_Name"])
    df["Vibe_ID"], vibe_uniques = pd.factorize(df["Vibe_Style"].fillna("Standard Reporting"))

    now_utc = pd.Timestamp.now(tz="UTC")
    df["Published_UTC"] = pd.to_datetime(df["Published"], errors="coerce", utc=True)
    df["Published_UTC"] = df["Published_UTC"].fillna(now_utc)
    hours_old = (now_utc - df["Published_UTC"]).dt.total_seconds() / 3600.0
    df["Recency_Bucket"] = hours_old.apply(recency_bucket).astype(int)

    df = assign_topic_clusters(df)

    df["Entity_Key"] = df.apply(
        lambda r: entity_key(
            str(r.get("Headline", "")),
            str(r.get("Summary", "")),
            str(r.get("Topic_Name", "")),
        ),
        axis=1,
    )
    df["Event_Key"] = df.apply(
        lambda r: event_key(
            str(r.get("Headline", "")),
            str(r.get("Summary", "")),
            str(r.get("Topic_Name", "")),
        ),
        axis=1,
    )
    df["Entity_ID"], entity_uniques = pd.factorize(df["Entity_Key"])
    df["Event_ID"], event_uniques = pd.factorize(df["Event_Key"])

    domain_freq = df["Topic_Name"].value_counts()
    df["Domain_Freq"] = df["Topic_Name"].map(domain_freq).astype(float)
    df["Popularity_Bucket"] = pd.qcut(
        df["Domain_Freq"].rank(method="first"), q=4, labels=False, duplicates="drop"
    ).astype(int)

    df["Item_ID"] = range(len(df))

    if len(df) < target_min:
        print(
            f"Warning: collected {len(df)} unique items, below target minimum {target_min}. "
            "Run again later to pick up new feed entries."
        )

    print(f"New scrape rows: {len(new_df)}")
    print(f"Merged unique rows: {len(df)}")
    print(
        "Cardinalities | "
        f"topics={len(topic_uniques)} vibes={len(vibe_uniques)} "
        f"clusters={df['Topic_Cluster_ID'].nunique()} entities={len(entity_uniques)} "
        f"events={len(event_uniques)}"
    )

    output_cols = [
        "Item_ID",
        "Headline",
        "Summary",
        "Link",
        "Published_UTC",
        "Topic_Name",
        "Topic_ID",
        "Emotion",
        "Vibe_Style",
        "Vibe_ID",
        "Topic_Cluster_ID",
        "Entity_Key",
        "Entity_ID",
        "Event_Key",
        "Event_ID",
        "Recency_Bucket",
        "Popularity_Bucket",
    ]
    df[output_cols].to_csv(data_path, index=False)
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSS/News scraper for training pool generation.")
    parser.add_argument("--mode", choices=["standard", "diversity"], default="standard")
    parser.add_argument("--target-min", type=int, default=TARGET_MIN)
    parser.add_argument("--target-max", type=int, default=TARGET_MAX)
    parser.add_argument("--use-google-news", action="store_true", default=USE_GOOGLE_NEWS)
    parser.add_argument("--max-google-queries", type=int, default=MAX_GOOGLE_QUERIES)
    parser.add_argument("--max-google-news-items", type=int, default=MAX_GOOGLE_NEWS_ITEMS)
    parser.add_argument("--slice-hours", type=int, default=GOOGLE_TIME_SLICE_HOURS)
    parser.add_argument("--slice-lookback-hours", type=int, default=GOOGLE_TIME_SLICE_LOOKBACK_HOURS)
    parser.add_argument("--fetch-mode", choices=["thread", "async"], default=DEFAULT_FETCH_MODE)
    parser.add_argument("--max-parallel-requests", type=int, default=MAX_PARALLEL_REQUESTS)
    args = parser.parse_args()
    main(
        mode=args.mode,
        target_min=args.target_min,
        target_max=args.target_max,
        use_google_news=args.use_google_news,
        max_google_queries=args.max_google_queries,
        max_google_news_items=args.max_google_news_items,
        slice_hours=args.slice_hours,
        slice_lookback_hours=args.slice_lookback_hours,
        fetch_mode=args.fetch_mode,
        max_parallel_requests=args.max_parallel_requests,
    )
