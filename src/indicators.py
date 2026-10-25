"""
Passive Security Indicators and URL Anomaly Detector for ScamShield.
Strict Invariant: ZERO network/HTTP requests or external socket calls.
All checks inspect syntax and domain structure offline.
"""

import re
from typing import List, Dict, Any
from urllib.parse import urlparse
import ipaddress

# URL regex matching standard protocols and bare domains
URL_REGEX = re.compile(
    r'(?:https?://[^\s<>"]+|www\.[^\s<>"]+|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s<>"]*)?)',
    re.IGNORECASE
)

# Known public URL shortening domains
KNOWN_SHORTENERS = {
    'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'is.gd', 'buff.ly',
    'ow.ly', 'cutt.ly', 'rb.gy', 'v.gd', 'rebrand.ly', 'tiny.cc'
}

# Coercive & Security Risk Rule Patterns
INDICATOR_RULES = [
    {
        "category": "Artificial Urgency & Coercion",
        "description": "Uses artificial deadlines or panic triggers to compel unthinking action.",
        "pattern": re.compile(
            r'\b(immediately|urgent|within\s+\d+\s*(?:hours?|mins?|minutes?)|account\s+(?:suspended|locked|blocked|terminated)|act\s+now|final\s+notice|expire[sd]?\s+(?:today|soon)|24\s*hours?)\b',
            re.IGNORECASE
        )
    },
    {
        "category": "Credential & Sensitive Data Harvesting",
        "description": "Prompts recipient for verification codes, passwords, PINs, or confidential IDs.",
        "pattern": re.compile(
            r'\b(otp|one[-\s]?time\s+password|pin\b|cvv|security\s+code|verify\s+(?:your\s+)?(?:identity|account|details|password)|login\s+to\s+verify|update\s+credentials|ssn|aadhaar)\b',
            re.IGNORECASE
        )
    },
    {
        "category": "Financial Lures & False Windfalls",
        "description": "Promises unexpected rewards, lotteries, refunds, or false billing alerts.",
        "pattern": re.compile(
            r'\b(won|winner|awarded|prize|lottery|claim\s+(?:your\s+)?(?:prize|reward|refund|cash)|tax\s+refund|compensation|crypto(?:currency)?\s+bonus|free\s+gift)\b',
            re.IGNORECASE
        )
    },
    {
        "category": "Authority & Brand Impersonation",
        "description": "Impersonates security desks, fraud prevention, or financial institutions.",
        "pattern": re.compile(
            r'\b(fraud\s+prevention|security\s+team|it\s+support|official\s+notice|bank\s+alert|customs\s+office|billing\s+department)\b',
            re.IGNORECASE
        )
    }
]

def extract_urls(text: str) -> List[str]:
    """Extracts all URL-like strings from text without network interactions."""
    raw_urls = URL_REGEX.findall(text)
    # Clean trailing punctuation commonly caught in natural text
    cleaned_urls = []
    for u in raw_urls:
        cleaned = u.rstrip('.,;:!?)>"\'')
        if len(cleaned) > 3 and '.' in cleaned:
            cleaned_urls.append(cleaned)
    return list(dict.fromkeys(cleaned_urls))

def analyze_url(raw_url: str) -> Dict[str, Any]:
    """
    Passively inspects an extracted URL for structural risk factors.
    Guaranteed zero external network calls.
    """
    has_scheme = raw_url.startswith(('http://', 'https://'))
    parse_target = raw_url if has_scheme else 'http://' + raw_url
    
    parsed = urlparse(parse_target)
    hostname = (parsed.hostname or '').lower()
    
    risk_flags = []
    
    # 1. Check if hostname is a raw IP address
    is_ip = False
    try:
        ipaddress.ip_address(hostname)
        is_ip = True
        risk_flags.append({
            "code": "RAW_IP_HOST",
            "label": "Raw IP Address Host",
            "detail": f"The link points directly to a numerical IP address ({hostname}) rather than a registered domain name, common in unverified hosting."
        })
    except ValueError:
        pass

    # 2. Check for URL shortener
    is_shortener = hostname in KNOWN_SHORTENERS
    if is_shortener:
        risk_flags.append({
            "code": "URL_SHORTENER",
            "label": "Shortened Link Destination",
            "detail": f"Uses '{hostname}' to mask the true destination web address."
        })

    # 3. Check for insecure HTTP scheme if explicitly stated
    is_insecure = parsed.scheme == 'http' and has_scheme
    if is_insecure and not is_ip:
        risk_flags.append({
            "code": "INSECURE_HTTP",
            "label": "Unencrypted HTTP Protocol",
            "detail": "Uses plaintext http:// instead of secure, encrypted https://."
        })

    # 4. Excessive subdomains (e.g. paypal.security.account-update.com)
    domain_parts = [p for p in hostname.split('.') if p]
    if len(domain_parts) >= 4 and not is_ip:
        risk_flags.append({
            "code": "EXCESSIVE_SUBDOMAINS",
            "label": "Multi-Layered Subdomains",
            "detail": f"Domain contains {len(domain_parts)} levels, a technique often used to disguise brand names in subdomains."
        })

    # 5. Non-standard port
    if parsed.port and parsed.port not in (80, 443):
        risk_flags.append({
            "code": "NON_STANDARD_PORT",
            "label": f"Non-Standard Port ({parsed.port})",
            "detail": f"Specifies an uncommon web port ({parsed.port}), which can bypass conventional web inspection."
        })

    return {
        "raw_url": raw_url,
        "hostname": hostname,
        "scheme": parsed.scheme if has_scheme else "none",
        "risk_flags": risk_flags,
        "is_suspicious": len(risk_flags) > 0
    }

def detect_security_indicators(text: str) -> List[Dict[str, Any]]:
    """Evaluates rule-based security patterns against text."""
    findings = []
    
    for rule in INDICATOR_RULES:
        matches = []
        for m in rule["pattern"].finditer(text):
            matches.append({
                "phrase": m.group(0),
                "start": m.start(),
                "end": m.end()
            })
            
        if matches:
            findings.append({
                "category": rule["category"],
                "description": rule["description"],
                "matches": matches
            })
            
    return findings
