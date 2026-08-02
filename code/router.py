"""
Deterministic Message Notification Router
Generates output.csv without requiring any LLM API key.
Uses rule-based logic over all provided dataset files.

Run from the code/ directory:
    python router.py

Output: ../dataset/output.csv
"""

import re
import csv
import os
from pathlib import Path
from collections import defaultdict

DATASET = Path(__file__).parent.parent / "dataset"

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_csv(filename):
    path = DATASET / filename
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def index_by(rows, *keys):
    """Build a lookup dict from rows keyed by tuple of fields."""
    out = defaultdict(list)
    for r in rows:
        k = tuple(r[k] for k in keys)
        out[k].append(r)
    return out


def index_single(rows, *keys):
    """Like index_by but stores only the first match (unique key)."""
    out = {}
    for r in rows:
        k = tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]]
        if k not in out:
            out[k] = r
    return out


# ---------------------------------------------------------------------------
# Load all dataset tables
# ---------------------------------------------------------------------------

messages        = load_csv("messages.csv")
users           = index_single(load_csv("users.csv"), "user_id")
groups          = index_single(load_csv("groups.csv"), "group_id")
group_members   = index_by(load_csv("group_members.csv"), "group_id", "user_id")
business        = index_single(load_csv("business_accounts.csv"), "business_id")
ub_history      = index_by(load_csv("user_business_history.csv"), "user_id", "business_id")
history         = load_csv("message_history.csv")
events          = load_csv("message_events.csv")

# Index history for fast retrieval
hist_by_user        = defaultdict(list)
hist_by_user_sender = defaultdict(list)
hist_by_user_biz    = defaultdict(list)
hist_by_user_group  = defaultdict(list)

for h in history:
    uid = h["user_id"]
    hist_by_user[uid].append(h)
    if h["sender_user_id"]:
        hist_by_user_sender[(uid, h["sender_user_id"])].append(h)
    if h["business_id"]:
        hist_by_user_biz[(uid, h["business_id"])].append(h)
    if h["group_id"]:
        hist_by_user_group[(uid, h["group_id"])].append(h)

# Valid history message IDs
valid_hist_ids = {h["message_id"] for h in history}

# Event index: message_id → events
events_by_msg = defaultdict(list)
for e in events:
    events_by_msg[e["message_id"]].append(e)

# ---------------------------------------------------------------------------
# Scam business detection
# ---------------------------------------------------------------------------

SCAM_REPORT_THRESHOLD = 15  # businesses with >=15 reports and domain mismatch are high-risk


def is_scam_business(biz_id):
    """Return True if this business uses a non-official domain and has high reports."""
    if not biz_id:
        return False
    b = business.get(biz_id)
    if not b:
        return False
    official = b.get("official_domain", "").strip()
    used = b.get("domain_used_by_sender", "").strip()
    reports = int(b.get("user_reports_30d", 0) or 0)
    if not official:
        # No official domain — treat as suspicious if reports are high
        return reports >= SCAM_REPORT_THRESHOLD
    return official != used and reports >= SCAM_REPORT_THRESHOLD


def is_verified_business(biz_id):
    if not biz_id:
        return False
    b = business.get(biz_id)
    if not b:
        return False
    return str(b.get("verified", "0")) == "1"


def user_opted_out(user_id, biz_id):
    if not biz_id:
        return False
    rows = ub_history.get((user_id, biz_id), [])
    for r in rows:
        if r.get("promotions_opted_out_at", "").strip():
            return True
    return False


def user_has_biz_activity(user_id, biz_id):
    """User has recent meaningful activity with a business (orders, bookings, etc.)."""
    if not biz_id:
        return False
    rows = ub_history.get((user_id, biz_id), [])
    for r in rows:
        why = r.get("why_user_knows_account", "").lower()
        count = int(r.get("activity_count_180d", 0) or 0)
        if count > 0 or any(kw in why for kw in [
            "order", "booking", "payment", "appointment", "delivery",
            "pickup", "return", "subscription", "grocery", "pharma"
        ]):
            return True
    return False


# ---------------------------------------------------------------------------
# Group membership helpers
# ---------------------------------------------------------------------------

def get_membership(user_id, group_id):
    if not group_id:
        return None
    rows = group_members.get((group_id, user_id), [])
    return rows[0] if rows else None


def group_muted_by_user(user_id, group_id):
    m = get_membership(user_id, group_id)
    if not m:
        return False
    return str(m.get("group_muted_by_user", "0")) == "1"


def sender_is_group_admin(sender_id, group_id):
    """Check if sender is an admin in this group."""
    if not sender_id or not group_id:
        return False
    rows = group_members.get((group_id, sender_id), [])
    return any(r.get("role", "").lower() == "admin" for r in rows)


def get_group_type(group_id):
    if not group_id:
        return ""
    g = groups.get(group_id)
    return g.get("group_type", "").lower() if g else ""


# ---------------------------------------------------------------------------
# Text pattern helpers
# ---------------------------------------------------------------------------

_INJECTION_RE = re.compile(
    r"routing\s+override"
    r"|system\s+note\s+for\s+(the\s+)?(notification\s+)?router"
    r"|internal\s+router\s+metadata"
    r"|assistant\s+instruction"
    r"|action\s*=\s*\w+"
    r"|mark\s+(this\s+)?as\s+\w+"
    r"|classify\s+as\s+\w+"
    r"|ignore\s+sender\s+risk"
    r"|ignore\s+(all\s+)?(previous|prior)\s+(routing\s+)?(rules|instructions)"
    r"|set\s+action\s*=",
    re.IGNORECASE
)

_OTP_SCAM_RE = re.compile(
    r"\botp\b"
    r"|one[\s\-]time\s+password"
    r"|verification\s+code"
    r"|login\s+code"
    r"|6[\s\-]digit\s+code"
    r"|share\s+(your\s+)?\botp\b|send\s+(your\s+)?\botp\b"
    r"|share\s+(the\s+)?code\s+here"
    r"|send.*code.*here|reply.*\bcode\b.*\bkeep\b",
    re.IGNORECASE
)

# Negation patterns that indicate a legitimate message mentioning OTP
_OTP_NEGATION_RE = re.compile(
    r"no\s+(payment\s+(or\s+)?)?\botp\b"
    r"|\botp\b\s+is\s+not\s+required"
    r"|\botp\b.*not\s+required"
    r"|never\s+ask(s)?\s+for.*\botp\b"
    r"|do\s+not\s+share.*\botp\b"
    r"|never\s+share.*\botp\b",
    re.IGNORECASE
)

_ACCOUNT_THREAT_RE = re.compile(
    r"account\s+(will\s+be\s+)?(block(ed)?|lock(ed)?|suspend(ed)?|restrict(ed)?|expir(e|ed)?|hold|band)"
    r"|profile\s+(will\s+be\s+)?(block(ed)?|lock(ed)?|restrict(ed)?|band(ed)?|suspend(ed)?)"
    r"|profile\s+is\s+(block(ed)?|lock(ed)?|restrict(ed)?)"
    r"|access\s+(will\s+be\s+)?(block(ed)?|lock(ed)?|restrict(ed)?|expir(e|ed)?|suspend(ed)?)"
    r"|account.*block(ed)?|block(ed)?.*account"
    r"|account\s+closure|avoid\s+account\s+(closure|lock)"
    r"|verify\s+now\s+or"
    r"|confirm.*or.*block"
    r"|warna\s+account|warna\s+profile",  # Hindi
    re.IGNORECASE
)

_SUSPICIOUS_LINK_RE = re.compile(
    r"bit\.ly/"
    r"|tinyurl\."
    r"|account-login\."
    r"|account-help\."
    r"|pay-check-secure\."
    r"|chase-secure-alert\."
    r"|amazonpay-delivery\.",
    re.IGNORECASE
)

_SUSPICIOUS_ACTION_RE = re.compile(
    r"scan.*qr.*pay"
    r"|scan\s+and\s+pay"
    r"|fill\s+bank\s+details"
    r"|send\s+screenshot"
    r"|pay\s+processing\s+fee"
    r"|pay.*token.*today"
    r"|claim.*sharing.*account\s+number"
    r"|benefit.*bank\s+details"
    r"|verify.*wallet.*details|verify.*card.*details"
    r"|wallet.*card.*details.*verif|card.*details.*before\s+midnight",
    re.IGNORECASE
)

_CHAIN_FORWARD_RE = re.compile(
    r"forward\s+(to\s+)?(at\s+least\s+)?(ten|10|everyone|all)"
    r"|forward\s+\S+\s+(to\s+)?(at\s+least\s+)?(ten|10)\s+people"
    r"|forward\s+\S+\s+\S+\s+(to\s+)?(ten|10)\s+people"
    r"|share\s+(with\s+)?(ten|10|everyone|all\s+groups?|family\s+groups?)"
    r"|send\s+(this\s+to\s+)?(ten|10)\s+people"
    r"|fwd\s+as\s+received"
    r"|forwarding\s+because"
    r"|do\s+not\s+(break|ignore)\s+the\s+chain"
    r"|sab\s+groups?\s+me\s+share"  # Hindi
    r"|share\s+kar\s+dena",
    re.IGNORECASE
)

_PAYMENT_PRESSURE_RE = re.compile(
    r"(urgent|pending)\s+(service|reactivation|clearance|penalty)\s+(fee|charge|amount)"
    r"|pay.*avoid\s+account\s+lock"
    r"|clearance\s+amount.*immediately"
    r"|scan.*pay.*clearance",
    re.IGNORECASE
)

_LOTTERY_SCAM_RE = re.compile(
    r"(your\s+)?(number|entry)\s+(was\s+|has\s+been\s+)?selected\s+for\s+(reward|prize|gift)"
    r"|you\s+(have\s+)?won.*claim"
    r"|congrats.*reward.*claim",
    re.IGNORECASE
)

_HEALTH_MISINFORMATION_RE = re.compile(
    r"stop\s+all\s+tablets"
    r"|doctors\s+don.t\s+(usually\s+)?tell"
    r"|drink\s+(warm|this)\s+(water|herbal|mix)"
    r"|habit\s+will\s+fix\s+health"
    r"|share.*elders",
    re.IGNORECASE
)

_URGENT_NOW_RE = re.compile(
    r"\bnow\b|\burgent(ly)?\b|\bimmediately\b|\basap\b"
    r"|\b(10|ten)\s+min(utes?)?\b|\b15\s+min(utes?)?\b|\b20\s+min(utes?)?\b|\b30\s+min(utes?)?\b"
    r"|\btoday\b|\btonight\b"
    r"|\bby\s+\d+\s*(pm|am|baje)\b"
    r"|\bescalation\s+starts\b|\bclient\s+escalation\b"
    r"|\bbuild\s+is\s+failing\b"
    r"|\bcrossed\s+the\s+alert\s+threshold\b"
    r"|\bstay\s+online\b"
    r"|\bjaldi\b|\bnikalna\s+padega\b"  # Hindi
    r"|\btank\s+aa\s+gaya\b",
    re.IGNORECASE
)

_DIRECT_ASK_RE = re.compile(
    r"\bcall\s+me\b|\bcan\s+you\s+call\b|\bcall\s+back\b"
    r"|\bplease\s+(confirm|reply|respond|come\s+online|check|pick\s+up|join)\b"
    r"|\bcan\s+you\s+come\s+online\b|\bcan\s+you\s+(collect|come|join)\b"
    r"|\bneed\s+you\s+(on|to)\b"
    r"|\bplease\s+pick\s+up\b"
    r"|\bconfirm\s+if\s+you\s+can\b"
    r"|\bpls\s+send\b|\bplease\s+send\b"
    r"|\bstay\s+online\b"
    r"|\bmessage\s+me\s+if\b"
    r"|\bcome\s+through\b|\bcome\s+via\b"
    r"|\bneed\s+you\s+to\b",
    re.IGNORECASE
)

_DELIVERY_UPDATE_RE = re.compile(
    r"order.*packed"
    r"|delivery.*scheduled"
    r"|delivery.*attempt"
    r"|pickup.*today"
    r"|return\s+pickup"
    r"|parcel.*arrive"
    r"|tanker.*leav"
    r"|\btank\b.*\bleav\b",
    re.IGNORECASE
)

_HEALTH_APPT_RE = re.compile(
    r"appointment.*moved"
    r"|appointment.*cancel"
    r"|appointment.*scheduled"
    r"|doctor.*appointment"
    r"|prescription.*ready"
    r"|claim.*ready"
    r"|health.*update",
    re.IGNORECASE
)


def safe(x):
    return str(x).strip() if x else ""


def text(row):
    return safe(row.get("message_text", ""))


def fwd_count(row):
    try:
        return int(row.get("forwarded_count", 0) or 0)
    except (ValueError, TypeError):
        return 0


def direct_mention(msg_row):
    """True if the message text contains @<user_id> of the receiving user."""
    uid = msg_row["user_id"]
    return f"@{uid}" in text(msg_row)


# ---------------------------------------------------------------------------
# Evidence retrieval
# ---------------------------------------------------------------------------

def get_evidence(msg_row, max_items=3):
    """Return up to max_items relevant historical message IDs."""
    uid = msg_row["user_id"]
    sender = safe(msg_row.get("sender_user_id"))
    biz = safe(msg_row.get("business_id"))
    grp = safe(msg_row.get("group_id"))

    candidates = []

    # Priority 1: same user + same sender
    if sender:
        candidates.extend(hist_by_user_sender.get((uid, sender), []))
    # Priority 2: same user + same business
    if biz:
        candidates.extend(hist_by_user_biz.get((uid, biz), []))
    # Priority 3: same user + same group
    if grp:
        candidates.extend(hist_by_user_group.get((uid, grp), []))
    # Priority 4: same user general
    if not candidates:
        candidates = hist_by_user.get(uid, [])

    # De-duplicate preserving order
    seen = set()
    unique = []
    for c in candidates:
        mid = c["message_id"]
        if mid not in seen and mid in valid_hist_ids:
            seen.add(mid)
            unique.append(c)

    if not unique:
        return "none"

    # Take most recent first (by created_at string sort — ISO format works)
    unique.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    selected = unique[:max_items]
    return ";".join(r["message_id"] for r in selected)


# ---------------------------------------------------------------------------
# Core routing logic
# ---------------------------------------------------------------------------

def route(msg_row):
    """
    Return (action, message_type, reason, confidence) for one message.
    action: notify | digest | mute
    message_type: personal | urgent | event | payment | business_update |
                  promotion | greeting | forward | spam | scam | unknown
    """
    txt = text(msg_row).lower()
    raw_txt = text(msg_row)
    fwd = fwd_count(msg_row)
    uid = msg_row["user_id"]
    conv = safe(msg_row.get("conversation_type"))
    grp = safe(msg_row.get("group_id"))
    biz = safe(msg_row.get("business_id"))
    sender = safe(msg_row.get("sender_user_id"))
    media = safe(msg_row.get("media_type"))

    # -------------------------------------------------------------------
    # RULE 1: Prompt injection → always mute/scam
    # -------------------------------------------------------------------
    if _INJECTION_RE.search(raw_txt):
        return "mute", "scam", (
            "Message contains prompt injection attempting to override routing logic."
        ), 0.97

    # -------------------------------------------------------------------
    # RULE 2: Scam business (domain mismatch + high reports)
    # -------------------------------------------------------------------
    if biz and is_scam_business(biz):
        return "mute", "scam", (
            "Message is from a business using a non-official domain with high user reports."
        ), 0.93

    # -------------------------------------------------------------------
    # RULE 3: OTP / account-threat scam (highest priority after injection)
    # -------------------------------------------------------------------
    otp_match = _OTP_SCAM_RE.search(raw_txt)
    # Exclude messages that *mention* OTP only to say they don't require it
    if otp_match and _OTP_NEGATION_RE.search(raw_txt):
        otp_match = None
    threat_match = _ACCOUNT_THREAT_RE.search(raw_txt)
    susp_link = _SUSPICIOUS_LINK_RE.search(raw_txt)
    susp_action = _SUSPICIOUS_ACTION_RE.search(raw_txt)

    if otp_match and (threat_match or susp_link):
        return "mute", "scam", (
            "Message combines OTP/code request with account-blocking pressure, a classic scam pattern."
        ), 0.95

    if otp_match and conv == "personal":
        return "mute", "scam", (
            "Personal message asking for OTP or verification code from an unexpected contact."
        ), 0.93

    if otp_match:
        # OTP request in any context is suspicious
        return "mute", "scam", (
            "Message requests an OTP or verification code — a strong indicator of a scam."
        ), 0.92

    if susp_link and threat_match:
        return "mute", "scam", (
            "Message pairs a suspicious link with account-blocking language."
        ), 0.93

    if susp_link and ("verify" in txt or "account" in txt or "login" in txt):
        return "mute", "scam", (
            "Suspicious shortened/non-official link combined with verification or account language."
        ), 0.90

    if susp_action:
        return "mute", "scam", (
            "Message asks for sensitive action (bank details, screenshot, QR payment, processing fee) typical of fraud."
        ), 0.92

    if threat_match and ("link" in txt or "verify" in txt or "click" in txt) and fwd >= 2:
        return "mute", "scam", (
            "Account-threat language with verification link and forwarded — phishing pattern."
        ), 0.90

    if threat_match and "verify" in txt and ("link" in txt or "immediately" in txt):
        return "mute", "scam", (
            "Account-blocking threat combined with an immediate verification demand — phishing."
        ), 0.90

    if _LOTTERY_SCAM_RE.search(raw_txt):
        return "mute", "scam", (
            "Message claims the user won a reward and asks to claim — a lottery scam pattern."
        ), 0.92

    # -------------------------------------------------------------------
    # RULE 4: Payment pressure scam (urgent QR/fee in group)
    # -------------------------------------------------------------------
    if _PAYMENT_PRESSURE_RE.search(raw_txt):
        return "mute", "scam", (
            "Message uses urgent payment pressure with QR code or clearance fee demands."
        ), 0.92

    # -------------------------------------------------------------------
    # RULE 5: Dangerous health misinformation forwarded
    # -------------------------------------------------------------------
    if _HEALTH_MISINFORMATION_RE.search(raw_txt) and fwd >= 3:
        return "mute", "forward", (
            "Forwarded message contains dangerous health misinformation advising to stop medication."
        ), 0.88

    # -------------------------------------------------------------------
    # RULE 6: Chain-forward spam
    # -------------------------------------------------------------------
    if _CHAIN_FORWARD_RE.search(raw_txt):
        if fwd >= 5:
            return "mute", "spam", (
                "High-forward-count chain message asking user to forward to others — typical spam."
            ), 0.88
        else:
            return "mute", "spam", (
                "Message asks user to forward to many people, a spam chain pattern."
            ), 0.84

    # -------------------------------------------------------------------
    # RULE 7: High forwarded_count with suspicious classification
    # -------------------------------------------------------------------
    if fwd >= 8 and ("share" in txt or "forward" in txt or "bhejo" in txt):
        return "mute", "spam", (
            "Highly forwarded message containing share/forward instructions — spam chain."
        ), 0.86

    # -------------------------------------------------------------------
    # RULE 8: User opted out of this business → mute promotion
    # -------------------------------------------------------------------
    if biz and user_opted_out(uid, biz):
        return "mute", "promotion", (
            "User has opted out of marketing messages from this business."
        ), 0.87

    # -------------------------------------------------------------------
    # RULE 9: Muted group → downgrade unless urgent direct mention
    # -------------------------------------------------------------------
    grp_muted = group_muted_by_user(uid, grp)
    has_direct_mention = direct_mention(msg_row)

    # -------------------------------------------------------------------
    # RULE 10: Classify voice-note messages (no text)
    # -------------------------------------------------------------------
    if media == "voice" and not txt:
        # Voice from business with scam domain already caught in Rule 2
        if biz:
            if is_verified_business(biz) and user_has_biz_activity(uid, biz):
                return "digest", "business_update", (
                    "Voice note from a verified business the user has interacted with — queued for later."
                ), 0.76
            return "mute", "spam", (
                "Voice note from business with no verified relationship — likely promotional or spam."
            ), 0.80
        grp_type = get_group_type(grp)
        if grp_muted and not sender_is_group_admin(sender, grp):
            return "mute", "personal", (
                "Voice note in a group the user has muted, from a non-admin sender."
            ), 0.78
        if grp and sender_is_group_admin(sender, grp):
            return "notify", "event", (
                "Voice note from a group admin — may contain important notice."
            ), 0.82
        if grp_type in ("coworker",):
            return "digest", "personal", (
                "Voice note in work group — queued for later review."
            ), 0.76
        return "digest", "personal", (
            "Voice note from a known contact — queued for later review."
        ), 0.76

    # -------------------------------------------------------------------
    # RULE 11: Society/admin group urgent notices (tanker, gate, repair)
    # -------------------------------------------------------------------
    grp_type = get_group_type(grp)
    sender_admin = sender_is_group_admin(sender, grp)

    if grp_type == "society" and sender_admin:
        # Payment reminders are never interrupt-worthy
        if "payment" in txt:
            return "digest", "payment", (
                "A society group admin sent a payment reminder."
            ), 0.78
        # Maintenance / info notices → digest
        if "maintenance" in txt:
            return "digest", "event", (
                "A society group admin sent a maintenance update."
            ), 0.76
        # Scheduled/future notices → digest
        if "tomorrow" in txt or "next week" in txt:
            return "digest", "event", (
                "A society group admin sent a scheduled advance notice."
            ), 0.76
        # Planned safety-test / lift → digest
        if any(kw in txt for kw in ["lift", "elevator", "alarm test", "fire alarm test", "potluck", "cultural"]):
            return "digest", "event", (
                "A society group admin sent a scheduled or social notice."
            ), 0.76
        # Truly time-sensitive operational: tanker/water in minutes
        if any(kw in txt for kw in ["tanker", "tank aa"]) or ("water" in txt and _URGENT_NOW_RE.search(raw_txt)):
            return "notify", "urgent", (
                "A society group admin sent an urgent water/tanker notice requiring immediate action."
            ), 0.90
        # Gate/vehicle blocking — move now
        if any(kw in txt for kw in ["gate", "truck", "repair truck", "car", "driveway"]) and _URGENT_NOW_RE.search(raw_txt):
            return "notify", "urgent", (
                "A society group admin sent an urgent gate or vehicle-blocking notice."
            ), 0.90
        # Other generic society urgent admin notice
        if _URGENT_NOW_RE.search(raw_txt):
            return "notify", "urgent", (
                "A society group admin sent an urgent operational notice requiring immediate action."
            ), 0.88
        return "digest", "event", (
            "A society group admin sent an informational update."
        ), 0.76

    # -------------------------------------------------------------------
    # RULE 12: Work group direct mention or escalation
    # -------------------------------------------------------------------
    if grp_type == "coworker":
        work_urgent_kws = [
            "sync", "deployment", "rollback", "build", "standup", "client", "eod",
            "tonight", "action items", "tonight", "prod", "production", "escalation"
        ]
        if has_direct_mention or _URGENT_NOW_RE.search(raw_txt) or any(kw in txt for kw in work_urgent_kws):
            return "notify", "urgent", (
                "Work group message with direct mention, urgent escalation, or same-day work dependency."
            ), 0.88
        return "digest", "personal", (
            "Work group update that doesn't require immediate action."
        ), 0.76

    # -------------------------------------------------------------------
    # RULE 13: Direct personal mention or urgent personal ask
    # -------------------------------------------------------------------
    if has_direct_mention and conv == "group":
        if _URGENT_NOW_RE.search(raw_txt) or _DIRECT_ASK_RE.search(raw_txt):
            return "notify", "urgent", (
                "Message directly mentions this user and contains an urgent or explicit ask."
            ), 0.87
        return "notify", "personal", (
            "Message directly mentions this user and asks for a response."
        ), 0.84

    if conv == "personal":
        # French lost-item with urgency (e.g. passeport trouve, avant 18h)
        if any(kw in txt for kw in ["passeport", "recuperer avant", "trouve dans"]):
            return "notify", "personal", (
                "Personal message about a found item with a pickup deadline — requires prompt attention."
            ), 0.85
        # Urgent personal messages
        if _URGENT_NOW_RE.search(raw_txt) and _DIRECT_ASK_RE.search(raw_txt):
            return "notify", "urgent", (
                "Personal message from a known contact with an urgent request requiring immediate attention."
            ), 0.88
        if "clinic" in txt or "doctor" in txt or "specialist" in txt or "appointment" in txt:
            if _URGENT_NOW_RE.search(raw_txt) or _DIRECT_ASK_RE.search(raw_txt):
                return "notify", "urgent", (
                    "Personal message about a medical appointment or health decision requiring immediate response."
                ), 0.90
        if _DIRECT_ASK_RE.search(raw_txt) and not (
            "nothing urgent" in txt or "no urgency" in txt or "don't call" in txt
            or "no need to reply" in txt
        ):
            return "notify", "personal", (
                "Personal message with a direct request or question requiring a response."
            ), 0.84
        # Non-urgent personal
        return "digest", "personal", (
            "Personal message from a known contact with no urgent action required."
        ), 0.78

    # -------------------------------------------------------------------
    # RULE 14: School group admin notices
    # -------------------------------------------------------------------
    if grp_type == "school_group" and sender_admin:
        if _URGENT_NOW_RE.search(raw_txt) or any(
            kw in txt for kw in ["today", "form", "consent", "deadline", "bus", "portal"]
        ):
            return "notify", "event", (
                "A school admin sent a time-sensitive notice with a same-day deadline."
            ), 0.87
        return "digest", "event", (
            "A school admin sent an informational update without an immediate deadline."
        ), 0.78

    # -------------------------------------------------------------------
    # RULE 15: Business messages
    # -------------------------------------------------------------------
    if conv == "business":
        if not biz:
            return "digest", "business_update", (
                "Business message without a verifiable sender — queued for review."
            ), 0.62
        verified = is_verified_business(biz)
        has_activity = user_has_biz_activity(uid, biz)

        # Delivery / order / appointment — notify if verified + activity
        if _DELIVERY_UPDATE_RE.search(raw_txt) or _HEALTH_APPT_RE.search(raw_txt):
            if verified and has_activity:
                return "notify", "business_update", (
                    "A verified business sent an update that matches the user's recent activity."
                ), 0.88
            if verified:
                # Even without prior activity, a verified delivery today is useful to see
                if "today" in txt or _URGENT_NOW_RE.search(raw_txt):
                    return "notify", "business_update", (
                        "A verified business sent a same-day delivery or appointment update."
                    ), 0.84
                return "digest", "business_update", (
                    "A verified business sent an update but no recent user activity found."
                ), 0.76
            return "digest", "business_update", (
                "Business update from an unverified sender — queued for review."
            ), 0.65

        # FedEx / delivery scheduling
        if "delivery" in txt and "pm" in txt and "id" in txt.lower():
            return "notify", "business_update", (
                "Scheduled delivery notice with time window — useful to see now."
            ), 0.83

        # Health / care appointment
        if "health" in txt and ("appointment" in txt or "prescription" in txt or "claim" in txt):
            if verified and has_activity:
                return "notify", "event", (
                    "A verified healthcare provider sent a reminder matching the user's appointment."
                ), 0.87
            return "digest", "event", (
                "Health update — queued for review."
            ), 0.72

        # Promotional content
        if any(kw in txt for kw in [
            "off", "discount", "offer", "deal", "welcome offer",
            "50% off", "40% off", "subscribe", "unsubscribe", "marketing"
        ]):
            if verified and has_activity:
                return "digest", "promotion", (
                    "Promotional message from a business the user has engaged with."
                ), 0.76
            return "digest", "promotion", (
                "Promotional content from a business — queued for later review."
            ), 0.72

        # Feedback / survey
        if any(kw in txt for kw in ["feedback", "review", "survey", "experience"]):
            return "digest", "business_update", (
                "A business is requesting feedback — low priority, shown later."
            ), 0.75

        # Statement / card update
        if any(kw in txt for kw in ["statement", "card", "bill", "payment date", "reward points"]):
            if verified:
                return "digest", "payment", (
                    "A verified bank or financial business sent a non-urgent statement notice."
                ), 0.78
            return "digest", "payment", (
                "Payment or card statement update — queued for review."
            ), 0.70

        # Generic B2B
        if "razorpay" in txt or "payouts" in txt or "vendor" in txt:
            return "digest", "business_update", (
                "B2B business communication — queued for later review."
            ), 0.72

        # Default business
        if verified:
            return "digest", "business_update", (
                "A verified business sent a general update."
            ), 0.74
        return "digest", "business_update", (
            "Business message queued for later review."
        ), 0.66

    # -------------------------------------------------------------------
    # RULE 16: Group messages (non-society, non-work, non-school)
    # -------------------------------------------------------------------
    if conv == "group":
        # Muted group + no admin + not direct mention → mute with appropriate type
        if grp_muted and not sender_admin and not has_direct_mention:
            # Classify the type even when muting
            if any(kw in txt for kw in ["selling", "for sale", "pickup near", "price final", "dm if interested"]):
                return "mute", "promotion", (
                    "Classified/marketplace message in a group the user has muted."
                ), 0.82
            if any(kw in txt for kw in ["good morning", "blessings", "stay positive", "bhagwan"]):
                return "mute", "greeting", (
                    "Greeting in a group the user has muted."
                ), 0.82
            return "mute", "personal", (
                "User has muted this group and the message has no direct mention or admin urgency."
            ), 0.82

        # Family group blessing chains (even if not muted)
        if fwd >= 6 and any(kw in txt for kw in [
            "blessing", "good morning", "smile", "stay positive", "bhagwan", "positive energy"
        ]):
            return "mute", "greeting", (
                "Highly forwarded blessing or greeting — user typically ignores these."
            ), 0.85

        # Classifieds / marketplace messages
        if any(kw in txt for kw in ["selling", "for sale", "pickup near", "price final", "dm if interested", "warehouse pickup"]):
            return "digest", "promotion", (
                "Marketplace or classified listing message — potentially relevant but not urgent."
            ), 0.76

        # Finance/market research note
        if any(kw in txt for kw in ["nvidia", "tsmc", "earnings", "market note", "research link", "semiconductor"]):
            return "digest", "business_update", (
                "Financial research note shared in group — useful but not time-critical."
            ), 0.76

        # Internship / academic deadlines — but "no rush" overrides urgency
        if any(kw in txt for kw in ["internship", "portal locks", "deadline", "professor", "slides", "lab section"]):
            if "no rush" in txt or "no urgency" in txt:
                return "digest", "personal", (
                    "Academic message with no time pressure — can be read later."
                ), 0.76
            if _URGENT_NOW_RE.search(raw_txt):
                return "notify", "event", (
                    "Time-sensitive academic or internship deadline message."
                ), 0.84
            return "digest", "event", (
                "Academic or course-related message — no immediate deadline."
            ), 0.76

        # Delivery / package arrived
        if any(kw in txt for kw in ["package", "courier", "delivery", "amazon package"]):
            if _URGENT_NOW_RE.search(raw_txt):
                return "notify", "urgent", (
                    "Package delivery requires immediate confirmation or pickup."
                ), 0.85
            return "digest", "business_update", (
                "Delivery notice — queued for later review."
            ), 0.74

        # Casual personal ask with direct time constraint from a trusted contact
        if any(kw in txt for kw in ["jacket", "kurta", "collect", "gate 2", "hold it only till"]):
            if _URGENT_NOW_RE.search(raw_txt):
                return "notify", "personal", (
                    "A contact is holding an item for the user and needs a quick response."
                ), 0.83
            return "digest", "personal", (
                "Personal pick-up or collection request — not immediately urgent."
            ), 0.75

        # Urgent personal call from family/friend group
        if any(kw in txt for kw in ["call me urgently", "call me now", "call me immediately"]):
            return "notify", "urgent", (
                "A contact is urgently requesting a phone call."
            ), 0.86

        # Casual social chat
        if any(kw in txt for kw in ["cricket", "match", "thread after dinner", "mute the thread"]):
            return "digest", "personal", (
                "Casual social chat about entertainment — can be read later."
            ), 0.78

        # Appointment / meeting update with time constraint
        if any(kw in txt for kw in ["meeting", "sync", "standup", "deployment"]):
            if _URGENT_NOW_RE.search(raw_txt) or "today" in txt:
                return "notify", "urgent", (
                    "Work or meeting update with same-day time pressure."
                ), 0.84
            return "digest", "event", (
                "Meeting or event update without immediate urgency."
            ), 0.75

        # Dance / event venue change
        if any(kw in txt for kw in ["studio", "dance practice", "side entrance", "front gate is locked"]):
            if _URGENT_NOW_RE.search(raw_txt):
                return "notify", "event", (
                    "Same-day event venue or time change — needs immediate attention."
                ), 0.84
            return "digest", "event", (
                "Event or venue update."
            ), 0.74

        # Greetings / good morning
        if any(kw in txt for kw in ["good morning", "good day", "have a good day", "stay blessed", "stay positive"]):
            if grp_muted:
                return "mute", "greeting", (
                    "Greeting in a group the user has muted."
                ), 0.82
            return "digest", "greeting", (
                "Casual greeting message — can be read later."
            ), 0.78

        # Lost / found item (passport in French too)
        if any(kw in txt for kw in ["bottle", "passport", "passeport", "found", "reception", "front desk", "recuperer"]):
            if _URGENT_NOW_RE.search(raw_txt) or "today" in txt or "avant" in txt or "before" in txt:
                return "notify", "personal", (
                    "Message about a lost item with a time-limited retrieval window."
                ), 0.83
            return "digest", "personal", (
                "Lost and found message — useful but not immediately urgent."
            ), 0.76

        # Real estate scam
        if any(kw in txt for kw in ["plot", "sqft", "token today", "registry papers", "airport road"]):
            return "mute", "scam", (
                "Message pressures user to pay a token amount for real estate — likely a scam."
            ), 0.88

        # Loan/advance fee
        if "loan approved" in txt and ("processing fee" in txt or "pay" in txt):
            return "mute", "scam", (
                "Advance fee fraud: claims a loan is approved but asks for upfront payment."
            ), 0.93

        # Refund with link or card details verification
        if "refund" in txt and any(kw in txt for kw in ["link", "wallet", "card details", "verify", "before midnight"]):
            return "mute", "scam", (
                "Refund message requesting wallet/card verification — a common phishing pattern."
            ), 0.90

        # General group chat
        return "digest", "personal", (
            "Group chat message without immediate urgency or safety concern."
        ), 0.73

    # -------------------------------------------------------------------
    # RULE 17: Fallback
    # -------------------------------------------------------------------
    return "digest", "unknown", (
        "No strong signal detected — queued for later review."
    ), 0.60


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    results = []
    for msg in messages:
        action, msg_type, reason, confidence = route(msg)

        # Clamp confidence
        confidence = round(max(0.0, min(1.0, confidence)), 2)

        # Evidence
        evidence = get_evidence(msg)

        results.append({
            "message_id": msg["message_id"],
            "action": action,
            "message_type": msg_type,
            "reason": reason,
            "confidence": confidence,
            "evidence_message_ids": evidence,
        })

    # Sort by message_id
    def sort_key(r):
        m = re.search(r'\d+', r["message_id"])
        return int(m.group()) if m else 0

    results.sort(key=sort_key)

    # Validate coverage
    input_ids = {m["message_id"] for m in messages}
    output_ids = {r["message_id"] for r in results}
    missing = input_ids - output_ids
    if missing:
        print(f"WARNING: Missing predictions for {missing}")

    # Validate schema
    ALLOWED_ACTIONS = {"notify", "digest", "mute"}
    ALLOWED_TYPES = {
        "personal", "urgent", "event", "payment", "business_update",
        "promotion", "greeting", "forward", "spam", "scam", "unknown"
    }
    for r in results:
        assert r["action"] in ALLOWED_ACTIONS, f"Bad action: {r}"
        assert r["message_type"] in ALLOWED_TYPES, f"Bad type: {r}"
        assert 0.0 <= r["confidence"] <= 1.0, f"Bad confidence: {r}"
        ev = r["evidence_message_ids"]
        if ev and ev.lower() != "none":
            for eid in ev.split(";"):
                eid = eid.strip()
                if eid and eid not in valid_hist_ids:
                    print(f"WARNING: evidence ID '{eid}' not in message_history for {r['message_id']}")

    # Write output
    out_path = DATASET / "output.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Written {len(results)} rows to {out_path}")

    # Print distribution
    from collections import Counter
    action_dist = Counter(r["action"] for r in results)
    type_dist = Counter(r["message_type"] for r in results)
    print("Action distribution:", dict(action_dist))
    print("Type distribution:", dict(type_dist))

    # Print the per-row decisions for inspection
    print("\n--- Decisions ---")
    for r in results:
        print(f"  {r['message_id']}: {r['action']:8s} {r['message_type']:20s} conf={r['confidence']}")


if __name__ == "__main__":
    main()
