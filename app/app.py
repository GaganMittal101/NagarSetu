
from __future__ import annotations

import csv
import io
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

REPORTS = ROOT / "data" / "reports.csv"
REPORT_IMAGES = ROOT / "data" / "report_images"

# Support the renamed NagarSetu model folder and the original trained-model
# location so the app continues to work after branding changes.
ROAD_MODEL_CANDIDATES = [
    ROOT / "models" / "runs" / "nagarsetu_road_damage_baseline" / "weights" / "best.pt",
    ROOT / "models" / "runs" / "civicfix_road_damage_baseline" / "weights" / "best.pt",
]
ROAD_MODEL = next(
    (candidate for candidate in ROAD_MODEL_CANDIDATES if candidate.exists()),
    ROAD_MODEL_CANDIDATES[0],
)

# ---------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------
NAVY = "#0B2550"
NAVY_2 = "#153C76"
BLUE = "#1769D1"
BLUE_DARK = "#0F57A7"
YELLOW = "#F5B719"
YELLOW_SOFT = "#FFF7DB"
BG = "#F5F7FB"
WHITE = "#FFFFFF"
TEXT = "#1D344E"
MUTED = "#6B7C91"
BORDER = "#DFE6EF"
GREEN = "#158A62"
RED = "#C53B3B"

# ---------------------------------------------------------------------
# Application data
# ---------------------------------------------------------------------
CATEGORIES = [
    "Pothole / Road Damage",
    "Garbage Overflow",
    "Broken Streetlight",
    "Water Leakage",
    "Traffic Signal Problem",
]

ROAD_DAMAGE_LABELS = {
    "Pothole": "Pothole",
    "Alligator Crack": "Alligator Crack",
    "Longitudinal Crack": "Longitudinal Crack",
    "Transverse Crack": "Transverse Crack",
}

REPORT_COLUMNS = [
    "report_id",
    "user_id",
    "user_name",
    "user_email",
    "created_at",
    "category",
    "description",
    "image_path",
    "city",
    "area",
    "latitude",
    "longitude",
    "ai_detections",
    "ai_confidence",
    "severity",
    "priority",
    "status",
    "department",
]

# ---------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------
def _read_rows_from_report_file() -> list[dict]:
    if not REPORTS.exists():
        return []

    try:
        with REPORTS.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception:
        return []


def ensure_store() -> None:
    REPORTS.parent.mkdir(parents=True, exist_ok=True)
    REPORT_IMAGES.mkdir(parents=True, exist_ok=True)

    if not REPORTS.exists():
        with REPORTS.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(REPORT_COLUMNS)
        return

    # Upgrade an older NagarSetu CSV to the current schema without
    # throwing away existing reports.
    try:
        with REPORTS.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_columns = reader.fieldnames or []
            old_rows = list(reader)

        if existing_columns != REPORT_COLUMNS:
            backup = REPORTS.with_suffix(".legacy.csv")
            if old_rows and not backup.exists():
                REPORTS.replace(backup)

                # The old rows are already in memory; recreate the active store
                # using the current schema without losing existing reports.
                with REPORTS.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
                    writer.writeheader()
                    for row in old_rows:
                        writer.writerow(
                            {column: row.get(column, "") for column in REPORT_COLUMNS}
                        )
            else:
                with REPORTS.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
                    writer.writeheader()
                    for row in old_rows:
                        writer.writerow(
                            {column: row.get(column, "") for column in REPORT_COLUMNS}
                        )
    except Exception:
        # If an unexpected file-format problem occurs, keep the app usable.
        pass


def load_reports() -> pd.DataFrame:
    ensure_store()

    try:
        df = pd.read_csv(REPORTS)
    except Exception:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    for column in REPORT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    return df[REPORT_COLUMNS]


def save_report(
    *,
    user_id: str,
    user_name: str,
    user_email: str,
    category: str,
    description: str,
    city: str,
    area: str,
    coords: str,
    image_bytes: bytes | None,
    image_suffix: str,
    ai_detections: str = "",
    ai_confidence: float | None = None,
) -> str:
    ensure_store()

    report_id = f"NS-{uuid.uuid4().hex[:8].upper()}"

    parts = [part.strip() for part in (coords or "").split(",")]
    latitude = parts[0] if len(parts) >= 1 else ""
    longitude = parts[1] if len(parts) >= 2 else ""

    image_path = ""
    if image_bytes:
        suffix = image_suffix.lower().replace(".", "")
        if suffix not in {"jpg", "jpeg", "png", "webp"}:
            suffix = "jpg"

        filename = f"{report_id}.{suffix}"
        destination = REPORT_IMAGES / filename
        destination.write_bytes(image_bytes)
        image_path = str(destination.relative_to(ROOT))

    row = {
        "report_id": report_id,
        "user_id": user_id,
        "user_name": user_name,
        "user_email": user_email,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category": category,
        "description": description.strip(),
        "image_path": image_path,
        "city": city.strip() or "Unknown",
        "area": area.strip() or "Unknown",
        "latitude": latitude,
        "longitude": longitude,
        "ai_detections": ai_detections,
        "ai_confidence": (
            f"{ai_confidence:.4f}" if ai_confidence is not None else ""
        ),
        "severity": "",
        "priority": "Unassessed",
        "status": "Open",
        "department": "Pending verification",
    }

    with REPORTS.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=REPORT_COLUMNS).writerow(row)

    return report_id



# ---------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------
USERS = ROOT / "data" / "users.csv"
USER_COLUMNS = [
    "user_id",
    "name",
    "email",
    "password_salt",
    "password_hash",
    "city",
    "created_at",
]

def ensure_users_store() -> None:
    USERS.parent.mkdir(parents=True, exist_ok=True)

    if not USERS.exists():
        with USERS.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=USER_COLUMNS)
            writer.writeheader()


def _clean_user_row(row: dict) -> dict:
    """Normalize CSV values without altering passwords."""
    cleaned = {}
    for column in USER_COLUMNS:
        value = row.get(column, "")
        if value is None:
            value = ""
        cleaned[column] = str(value).strip() if column != "password_hash" else str(value)
        if column == "password_salt":
            cleaned[column] = str(value).strip()
        elif column == "email":
            cleaned[column] = str(value).strip().lower()
        elif column == "name":
            cleaned[column] = str(value).strip()
        elif column == "city":
            cleaned[column] = str(value).strip()
    return cleaned


def load_user_records() -> list[dict]:
    """Read users directly from CSV so authentication does not depend on pandas."""
    ensure_users_store()

    try:
        with USERS.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return [
                _clean_user_row(row)
                for row in reader
                if any(str(value or "").strip() for value in row.values())
            ]
    except (OSError, csv.Error):
        return []


def load_users() -> pd.DataFrame:
    records = load_user_records()
    if not records:
        return pd.DataFrame(columns=USER_COLUMNS)
    return pd.DataFrame(records, columns=USER_COLUMNS)


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    import hashlib
    import secrets

    if not isinstance(password, str):
        password = str(password)

    if salt is None:
        salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        220_000,
    ).hex()

    return salt.hex(), password_hash


def verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    import hashlib
    import hmac

    if not password or not salt_hex or not expected_hash:
        return False

    try:
        salt = bytes.fromhex(str(salt_hex).strip())

        actual_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            220_000,
        ).hex()

        return hmac.compare_digest(
            actual_hash,
            str(expected_hash).strip(),
        )
    except (ValueError, TypeError):
        return False


def register_user(
    name: str,
    email: str,
    password: str,
    city: str,
) -> tuple[bool, str]:
    name = name.strip()
    email = email.strip().lower()
    city = city.strip() or "Unknown"

    if len(name) < 2:
        return False, "Please enter your full name."

    # Normalize common browser/autofill whitespace and validate the full address.
    email = email.strip().replace("\u200b", "")
    email = re.sub(r"\s+", "", email)
    email_pattern = r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$"
    if not re.fullmatch(email_pattern, email, flags=re.IGNORECASE):
        return False, "Please enter a valid email address."

    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    existing_records = load_user_records()

    if any(
        row.get("email", "").strip().lower() == email
        for row in existing_records
    ):
        return False, "An account with this email already exists."

    salt_hex, password_hash = hash_password(password)

    row = {
        "user_id": f"U-{uuid.uuid4().hex[:8].upper()}",
        "name": name,
        "email": email,
        "password_salt": salt_hex,
        "password_hash": password_hash,
        "city": city,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        with USERS.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=USER_COLUMNS)
            writer.writerow(row)
    except OSError as exc:
        return False, f"Could not save account: {exc}"

    # Verify the newly saved credentials immediately. This catches file/encoding
    # problems at creation time instead of failing later at sign-in.
    saved_records = load_user_records()
    saved_user = next(
        (
            record
            for record in saved_records
            if record["email"] == email
        ),
        None,
    )

    if not saved_user or not verify_password(
        password,
        saved_user["password_salt"],
        saved_user["password_hash"],
    ):
        return False, "Account was saved, but credential verification failed."

    return True, "Account created successfully."


def authenticate_user(email: str, password: str) -> dict | None:
    email = re.sub(r"\s+", "", (email or "").replace("\u200b", "")).strip().lower()

    if not email or not password:
        return None

    for user in load_user_records():
        if user["email"] != email:
            continue

        if verify_password(
            password,
            user["password_salt"],
            user["password_hash"],
        ):
            return user

        return None

    return None


def render_auth_page() -> None:
    """Professional login/create-account screen without nested Markdown HTML blocks."""

    st.markdown(
        """
<style>
.auth-page-spacer{height:18px}
.auth-left-panel{
    background:linear-gradient(145deg,#0B2550 0%,#123A73 55%,#173F7A 100%);
    border-radius:28px;
    min-height:700px;
    padding:52px 52px 42px 52px;
    box-shadow:0 24px 60px rgba(10,36,78,.16);
    box-sizing:border-box;
}
.auth-brand-row{display:flex;align-items:center;gap:12px}
.auth-brand-icon{
    width:48px;height:48px;border-radius:14px;
    background:#F5B719;color:#0B2550;
    display:flex;align-items:center;justify-content:center;
    font-size:24px;font-weight:900
}
.auth-brand-name{color:#fff;font-size:21px;font-weight:800;line-height:1}
.auth-brand-sub{color:#BFD0E6;font-size:12px;margin-top:5px}
.auth-kicker{color:#FFDA65;font-size:11px;font-weight:800;letter-spacing:2.1px;text-transform:uppercase}
.auth-title{color:#fff;font-size:clamp(3rem,4.5vw,5rem);font-weight:800;line-height:.98;letter-spacing:-.055em;margin-top:16px}
.auth-copy{color:#DCE8FA;font-size:16px;line-height:1.65;max-width:650px;margin-top:20px}
.auth-point{display:flex;align-items:center;gap:10px;color:#F7FAFE;font-size:14px;line-height:1.4;margin-top:13px}
.auth-check{
    width:26px;height:26px;min-width:26px;border-radius:50%;
    background:rgba(245,183,25,.16);color:#FFD45A;
    display:inline-flex;align-items:center;justify-content:center;
    font-weight:900
}
.auth-trust{
    margin-top:28px;color:#BFD0E6;font-size:12px;line-height:1.65;max-width:650px
}
.auth-trust strong{color:#fff}
.auth-foot{margin-top:70px;color:#8FA6C5;font-size:11px}
.auth-right-panel{padding:35px 30px 30px 18px}
.auth-panel-title{color:#0B2550;font-size:36px;line-height:1.05;font-weight:800;letter-spacing:-.035em}
.auth-panel-copy{color:#6B7C91;font-size:15px;line-height:1.6;margin-top:7px;margin-bottom:22px}
@media(max-width:900px){
    .auth-left-panel{min-height:520px;padding:36px}
    .auth-right-panel{padding:24px 6px}
    .auth-title{font-size:3rem}
}
</style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.02, 0.98], gap="large")

    with left:
        # IMPORTANT: build the HTML without leading indentation/newlines.
        # This prevents Streamlit Markdown from treating it as a code block.
        left_html = "".join(
            [
                "<div class='auth-left-panel'>",
                "<div class='auth-brand-row'>",
                "<div class='auth-brand-icon'>C</div>",
                "<div>",
                "<div class='auth-brand-name'>NagarSetu</div>",
                "<div class='auth-brand-sub'>Predictive civic intelligence</div>",
                "</div>",
                "</div>",
                "<div style='height:115px'></div>",
                "<div class='auth-kicker'>CIVIC INTELLIGENCE PLATFORM</div>",
                "<div class='auth-title'>Turn civic problems<br>into action.</div>",
                "<div class='auth-copy'>"
                "Report local issues with photo, description and location. "
                "NagarSetu helps analyse supported road damage, organize evidence, "
                "and keep every report traceable from submission to follow-up."
                "</div>",
                "<div style='height:10px'></div>",
                "<div class='auth-point'><span class='auth-check'>✓</span><span>Evidence-first civic reporting</span></div>",
                "<div class='auth-point'><span class='auth-check'>✓</span><span>Real AI-assisted road-damage analysis</span></div>",
                "<div class='auth-point'><span class='auth-check'>✓</span><span>Track submitted civic cases</span></div>",
                "<div class='auth-trust'><strong>Built for trust.</strong><br>"
                "Model confidence is kept separate from severity, priority and final civic action. "
                "Unknown information stays unknown until it is validated."
                "</div>",
                "<div class='auth-foot'>NagarSetu • Evidence • Transparency • Verification</div>",
                "</div>",
            ]
        )
        st.markdown(left_html, unsafe_allow_html=True)

    with right:
        st.markdown("<div class='auth-right-panel'>", unsafe_allow_html=True)

        # Apply a requested mode BEFORE the auth_mode widget is instantiated.
        if st.session_state.pending_auth_mode in {"Sign In", "Create Account"}:
            st.session_state.auth_mode = st.session_state.pending_auth_mode
            st.session_state.pending_auth_mode = None

        mode = st.radio(
            "Access",
            ["Sign In", "Create Account"],
            horizontal=True,
            key="auth_mode",
        )

        if mode == "Sign In":
            if "auth_notice" in st.session_state:
                st.success(st.session_state.auth_notice)
                del st.session_state.auth_notice

            st.markdown(
                "<div class='auth-panel-title'>Welcome back</div>"
                "<div class='auth-panel-copy'>Sign in to continue with NagarSetu.</div>",
                unsafe_allow_html=True,
            )

            email = st.text_input(
                "Email",
                placeholder="you@example.com",
                key="login_email",
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password",
            )

            if st.button(
                "Sign In",
                key="login_submit",
                width="stretch",
            ):
                user = authenticate_user(email, password)

                if user:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.session_state.page = "Home"
                    st.rerun()
                else:
                    normalized_login_email = re.sub(
                        r"\s+",
                        "",
                        (email or "").replace("\u200b", ""),
                    ).strip().lower()

                    account_exists = any(
                        record["email"] == normalized_login_email
                        for record in load_user_records()
                    )

                    if not account_exists:
                        st.error(
                            "No account found for this email. Create an account first."
                        )
                    else:
                        st.error("Password is incorrect. Check the password and try again.")


        else:
            st.markdown(
                "<div class='auth-panel-title'>Create your account</div>"
                "<div class='auth-panel-copy'>Create an account to keep your NagarSetu reports together.</div>",
                unsafe_allow_html=True,
            )

            name = st.text_input(
                "Full name",
                placeholder="Your full name",
                key="register_name",
            )
            email = st.text_input(
                "Email",
                placeholder="you@example.com",
                key="register_email",
            )
            st.caption("Use a valid address such as name@example.com.")
            password = st.text_input(
                "Password",
                type="password",
                placeholder="At least 8 characters",
                key="register_password",
            )
            confirm = st.text_input(
                "Confirm password",
                type="password",
                placeholder="Repeat your password",
                key="register_confirm",
            )
            city = st.text_input(
                "City",
                placeholder="Jaipur",
                key="register_city",
            )

            if st.button(
                "Create Account",
                key="register_submit",
                width="stretch",
            ):
                if password != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, message = register_user(
                        name,
                        email,
                        password,
                        city,
                    )

                    if ok:
                        # Sign the newly created account in immediately. This avoids
                        # a fragile "create -> sign in again" transition and confirms
                        # the saved credentials are actually usable.
                        created_user = authenticate_user(email, password)
                        if created_user:
                            st.session_state.authenticated = True
                            st.session_state.user = created_user
                            st.session_state.page = "Home"
                            st.session_state.analysis = None
                            st.session_state.submitted_report_id = None
                            st.rerun()
                        else:
                            st.session_state.login_email = email.strip().lower()
                            st.session_state.login_password = ""
                            st.session_state.pending_auth_mode = "Sign In"
                            st.session_state.auth_notice = (
                                "Account created successfully. Please sign in with your new password."
                            )
                            st.rerun()
                    else:
                        st.error(message)

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# AI model helpers
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_road_model():
    if not ROAD_MODEL.exists():
        return None

    try:
        from ultralytics import YOLO

        return YOLO(str(ROAD_MODEL))
    except Exception:
        return None


def run_road_analysis(image: Image.Image, confidence_threshold: float):
    model = load_road_model()
    if model is None:
        return None, None, "Road-damage model could not be loaded."

    try:
        results = model.predict(
            source=image,
            imgsz=640,
            conf=confidence_threshold,
            device="cpu",
            verbose=False,
            max_det=50,
        )

        if not results:
            return None, None, "The model returned no prediction result."

        result = results[0]

        detections: list[dict] = []
        names = result.names

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = str(names[class_id])

                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                    }
                )

        annotated = result.plot()
        annotated_rgb = annotated[:, :, ::-1]

        return detections, annotated_rgb, None

    except Exception as exc:
        return None, None, f"AI analysis failed: {exc}"


def format_detection_summary(detections: list[dict]) -> str:
    if not detections:
        return ""

    counts = Counter(item["label"] for item in detections)
    parts = []

    for label, count in counts.most_common():
        parts.append(f"{label} × {count}")

    return ", ".join(parts)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "submitted_report_id" not in st.session_state:
    st.session_state.submitted_report_id = None

if "draft_image_bytes" not in st.session_state:
    st.session_state.draft_image_bytes = None

if "draft_image_suffix" not in st.session_state:
    st.session_state.draft_image_suffix = "jpg"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user" not in st.session_state:
    st.session_state.user = None

if "pending_auth_mode" not in st.session_state:
    st.session_state.pending_auth_mode = None


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="NagarSetu",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Professional UI CSS
# ---------------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
    --navy: {NAVY};
    --navy2: {NAVY_2};
    --blue: {BLUE};
    --yellow: {YELLOW};
    --bg: {BG};
    --text: {TEXT};
    --muted: {MUTED};
    --border: {BORDER};
}}

html, body, [class*="css"] {{
    font-family: "Inter", sans-serif;
}}

.stApp {{
    background: var(--bg);
    color: var(--text);
}}

.block-container {{
    width: 100%;
    max-width: 1520px;
    padding-top: 1.4rem !important;
    padding-right: 2.5rem !important;
    padding-left: 2.5rem !important;
    padding-bottom: 2.5rem !important;
}}

#MainMenu,
footer {{
    visibility: hidden;
}}

header[data-testid="stHeader"] {{
    background: transparent !important;
}}

div[data-testid="stToolbar"] {{
    right: 1rem;
}}

/* ---------------- Sidebar ---------------- */

section[data-testid="stSidebar"] {{
    width: 310px !important;
    background: var(--navy) !important;
}}

section[data-testid="stSidebar"] > div {{
    width: 310px !important;
    background: var(--navy) !important;
}}

section[data-testid="stSidebar"] * {{
    color: white;
}}

.sidebar-brand {{
    padding: .5rem .6rem 1.4rem;
    border-bottom: 1px solid rgba(255,255,255,.10);
    margin-bottom: 1.1rem;
}}

.brand-row {{
    display: flex;
    align-items: center;
    gap: .8rem;
}}

.brand-icon {{
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: var(--yellow);
    color: var(--navy);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: 900;
}}

.brand-name {{
    font-size: 20px;
    line-height: 1;
    font-weight: 800;
}}

.brand-sub {{
    margin-top: 5px;
    font-size: 12px;
    color: #B9C9DF !important;
}}

section[data-testid="stSidebar"] [data-testid="stRadio"] > label {{
    display: none;
}}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {{
    gap: 6px;
}}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {{
    border-radius: 12px;
    padding: 10px 12px;
    transition: .15s ease;
}}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
    background: rgba(255,255,255,.08);
}}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label p {{
    font-size: 14px;
    font-weight: 600;
    color: #E8EEF8 !important;
}}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {{
    background: rgba(255,255,255,.11);
    box-shadow: inset 4px 0 0 var(--yellow);
}}

.sidebar-status {{
    margin-top: 1.6rem;
    padding: 14px;
    border-radius: 14px;
    background: rgba(255,255,255,.055);
    border: 1px solid rgba(255,255,255,.09);
}}

.status-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    font-weight: 800;
    color: #8FA6C5 !important;
}}

.status-title {{
    margin-top: 6px;
    font-size: 13px;
    font-weight: 800;
}}

.status-copy {{
    margin-top: 5px;
    font-size: 11px;
    line-height: 1.55;
    color: #B9C9DF !important;
}}

/* ---------------- Top bar ---------------- */

.topbar {{
    background: rgba(255,255,255,.94);
    border: 1px solid var(--border);
    border-radius: 15px;
    min-height: 58px;
    padding: 0 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 6px 22px rgba(15,42,76,.035);
    margin-bottom: 20px;
}}

.context {{
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 800;
}}

.context strong {{
    color: var(--navy);
    margin-left: 6px;
}}

.profile {{
    color: var(--navy);
    font-size: 12px;
    font-weight: 700;
}}

.avatar {{
    display: inline-flex;
    width: 31px;
    height: 31px;
    border-radius: 50%;
    align-items: center;
    justify-content: center;
    background: var(--yellow);
    color: var(--navy);
    font-size: 11px;
    font-weight: 900;
    margin-right: 7px;
}}

/* ---------------- Hero ---------------- */

.hero {{
    background:
        radial-gradient(circle at 86% 22%, rgba(245,183,25,.16), transparent 23%),
        linear-gradient(125deg, var(--navy), var(--navy2));
    color: white;
    border-radius: 28px;
    padding: 58px 68px;
    min-height: 455px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow: visible;
    box-shadow: 0 20px 48px rgba(10,36,78,.12);
}}

.hero-kicker {{
    color: #FFDA65;
    font-size: 12px;
    letter-spacing: 2.4px;
    font-weight: 800;
    text-transform: uppercase;
}}

.hero-title {{
    margin-top: 17px;
    max-width: 920px;
    color: #fff;
    font-size: clamp(3.15rem, 4.8vw, 5.3rem);
    line-height: .98;
    font-weight: 800;
    letter-spacing: -0.055em;
}}

.hero-copy {{
    margin-top: 24px;
    max-width: 780px;
    color: #DCE8FA;
    font-size: 17px;
    line-height: 1.65;
}}

.hero-note {{
    margin-top: 22px;
    color: #BFD0E6;
    font-size: 12px;
    line-height: 1.5;
}}

.hero-actions {{
    margin-top: 30px;
}}

/* ---------------- Sections ---------------- */

.section-kicker {{
    margin-top: 26px;
    color: var(--blue);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    font-weight: 800;
}}

.section-title {{
    margin-top: 5px;
    color: var(--navy);
    font-size: 30px;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -.03em;
}}

.section-copy {{
    margin-top: 8px;
    margin-bottom: 18px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.65;
    max-width: 900px;
}}

/* ---------------- Cards ---------------- */

.card,
.feature,
.step,
.result,
.trust,
.insight-card {{
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 18px;
    box-shadow: 0 8px 28px rgba(15,42,76,.045);
}}

.card {{
    padding: 24px;
}}

.feature {{
    padding: 21px;
    min-height: 168px;
}}

.feature-no {{
    color: var(--blue);
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 11px;
}}

.feature-title {{
    color: var(--navy);
    font-size: 16px;
    font-weight: 800;
}}

.feature-copy {{
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
    margin-top: 7px;
}}

.step {{
    padding: 20px;
    min-height: 145px;
}}

.step-no {{
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--yellow-soft);
    color: #735000;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 900;
    margin-bottom: 12px;
}}

.step-title {{
    color: var(--navy);
    font-size: 15px;
    font-weight: 800;
}}

.step-copy {{
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
    margin-top: 6px;
}}

/* ---------------- KPI ---------------- */

.kpi-card {{
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px;
    min-height: 122px;
    box-shadow: 0 8px 28px rgba(15,42,76,.04);
}}

.kpi-label {{
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
}}

.kpi-value {{
    color: var(--navy);
    font-size: 32px;
    line-height: 1;
    font-weight: 800;
    margin-top: 10px;
}}

.kpi-meta {{
    color: var(--green);
    font-size: 11px;
    font-weight: 700;
    margin-top: 8px;
}}

/* ---------------- Results ---------------- */

.result {{
    padding: 24px;
}}

.result-label {{
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
}}

.result-title {{
    color: var(--navy);
    font-size: 25px;
    line-height: 1.2;
    font-weight: 800;
    margin-top: 5px;
}}

.info-row {{
    display: flex;
    justify-content: space-between;
    gap: 18px;
    padding: 11px 0;
    border-bottom: 1px solid #EEF2F7;
}}

.info-key {{
    color: var(--muted);
    font-size: 12px;
}}

.info-val {{
    color: var(--navy);
    font-size: 12px;
    font-weight: 800;
    text-align: right;
}}

.badge {{
    display: inline-block;
    padding: 6px 9px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 800;
}}

.badge-blue {{
    background: #EDF5FF;
    color: var(--blue);
}}

.badge-green {{
    background: #EAF8F2;
    color: var(--green);
}}

.badge-yellow {{
    background: var(--yellow-soft);
    color: #735000;
}}

.badge-red {{
    background: #FFF0F0;
    color: var(--red);
}}

/* ---------------- Notices ---------------- */

.notice {{
    padding: 14px 16px;
    background: var(--yellow-soft);
    border: 1px solid #FFDA70;
    border-left: 5px solid var(--yellow);
    border-radius: 14px;
}}

.notice-title {{
    color: #735000;
    font-size: 12px;
    font-weight: 800;
}}

.notice-copy {{
    color: #7D6935;
    font-size: 11px;
    line-height: 1.55;
    margin-top: 4px;
}}

/* ---------------- Streamlit widgets ---------------- */

div[data-testid="stButton"] > button {{
    min-height: 48px;
    border-radius: 12px !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    border: 1px solid var(--blue) !important;
    background: var(--blue) !important;
    color: white !important;
    transition: .15s ease;
}}

div[data-testid="stButton"] > button:hover {{
    background: var(--blue-dark) !important;
    border-color: var(--blue-dark) !important;
    color: white !important;
}}

div[data-testid="stFileUploader"] section {{
    border: 1.5px dashed #9EB8D4 !important;
    border-radius: 15px !important;
    background: #FBFDFF !important;
}}

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-baseweb="select"] > div {{
    border-radius: 12px !important;
}}

div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stFileUploader"] label {{
    font-size: 13px !important;
    font-weight: 700 !important;
}}

div[data-testid="stMetric"] {{
    background: white;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 10px 14px;
}}

.stDataFrame {{
    border-radius: 14px;
}}

/* Mobile / narrow screens */
@media (max-width: 900px) {{
    .block-container {{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}

    .hero {{
        padding: 42px 28px;
        min-height: 430px;
    }}

    .hero-title {{
        font-size: clamp(2.5rem, 11vw, 4rem);
    }}

    .hero-copy {{
        font-size: 15px;
    }}

    section[data-testid="stSidebar"] {{
        width: 280px !important;
    }}
}}
</style>
""",
    unsafe_allow_html=True,
)

# Authentication gate.
if not st.session_state.authenticated:
    render_auth_page()
    st.stop()

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-row">
                <div class="brand-icon">C</div>
                <div>
                    <div class="brand-name">NagarSetu</div>
                    <div class="brand-sub">Predictive civic intelligence</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        "Home",
        "Report Problem",
        "My Reports",
        "City Insights",
        "AI & Trust",
        "About",
    ]

    icons = {
        "Home": "⌂",
        "Report Problem": "＋",
        "My Reports": "▤",
        "City Insights": "◈",
        "AI & Trust": "◉",
        "About": "ⓘ",
    }

    options = [f"{icons[p]}   {p}" for p in pages]
    current_index = pages.index(st.session_state.page)

    selected = st.radio(
        "Workspace",
        options,
        index=current_index,
        label_visibility="collapsed",
    )

    selected_page = next(
        page for page in pages if selected.endswith(page)
    )

    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.session_state.analysis = None
        st.session_state.submitted_report_id = None
        st.rerun()

    model_available = ROAD_MODEL.exists()
    model_status = (
        "AI model connected"
        if model_available
        else "AI model unavailable"
    )
    model_copy = (
        "Road-damage detection is available for uploaded images."
        if model_available
        else "The interface works, but the road-damage model file was not found."
    )

    st.markdown(
        f"""
        <div class="sidebar-status">
            <div class="status-label">System status</div>
            <div class="status-title">
                {'●' if model_available else '○'} {model_status}
            </div>
            <div class="status-copy">{model_copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_user = st.session_state.user or {}
    current_name = str(current_user.get("name", "User"))
    current_city = str(current_user.get("city", "Unknown"))
    avatar = current_name[:1].upper() if current_name else "G"

    st.markdown(
        f"""
        <div style="
            margin-top:12px;
            padding:12px;
            border-radius:14px;
            background:rgba(255,255,255,.045);
            border:1px solid rgba(255,255,255,.08);
        ">
            <div style="
                color:#B9C9DF !important;
                font-size:10px;
                text-transform:uppercase;
                letter-spacing:1px;
                font-weight:800;
            ">Signed in as</div>
            <div style="
                color:#fff !important;
                font-size:13px;
                font-weight:800;
                margin-top:4px;
            ">{current_name}</div>
            <div style="
                color:#8FA6C5 !important;
                font-size:10px;
                margin-top:3px;
            ">{current_city}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "↪  Log out",
        key="logout",
        width="stretch",
    ):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.analysis = None
        st.session_state.submitted_report_id = None
        st.rerun()

# ---------------------------------------------------------------------
# Top navigation / context bar
# ---------------------------------------------------------------------
st.markdown(
    f"""
    <div class="topbar">
        <div class="context">
            NAGARSETU <strong>/ {st.session_state.page}</strong>
        </div>
        <div class="profile">
            <span class="avatar">{str((st.session_state.user or {}).get("name", "Guest"))[:1].upper()}</span>
            {str((st.session_state.user or {}).get("name", "Guest"))}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------
if st.session_state.page == "Home":
    st.markdown(
        "<div class='hero'>"
        "<div class='hero-kicker'>CIVIC INTELLIGENCE PLATFORM</div>"
        "<div class='hero-title'>See a civic problem.<br>Turn it into action.</div>"
        "<div class='hero-copy'>"
        "Report local issues with photo, description and location. "
        "NagarSetu helps analyse supported road damage, organize evidence "
        "and keep every report traceable from submission to follow-up."
        "</div>"
        "<div class='hero-note'>Evidence first · AI-assisted · Human verification</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    action_a, action_b = st.columns(2, gap="medium")

    with action_a:
        if st.button(
            "＋  Report a civic problem",
            key="home_report",
            width="stretch",
        ):
            st.session_state.page = "Report Problem"
            st.rerun()

    with action_b:
        if st.button(
            "◈  Explore city insights",
            key="home_insights",
            width="stretch",
        ):
            st.session_state.page = "City Insights"
            st.rerun()

    st.markdown(
        '<div style="height:22px"></div>',
        unsafe_allow_html=True,
    )

    df = load_reports()

    total_reports = len(df)
    open_cases = (
        int((df["status"] == "Open").sum()) if not df.empty else 0
    )
    resolved_cases = (
        int((df["status"] == "Resolved").sum()) if not df.empty else 0
    )
    ai_cases = (
        int(df["ai_detections"].fillna("").astype(str).str.strip().ne("").sum())
        if not df.empty
        else 0
    )

    stats = [
        ("Reports", total_reports, "Submitted cases"),
        ("Open cases", open_cases, "Currently active"),
        ("AI analysed", ai_cases, "Saved with detections"),
        ("Resolved", resolved_cases, "Closed cases"),
    ]

    cols = st.columns(4, gap="medium")

    for col, (label, value, meta) in zip(cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-meta">{meta}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-kicker">Designed for trust</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Useful information without fake certainty</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="section-copy">
            NagarSetu separates what the system can detect from what still needs
            verification. AI confidence is not the same thing as severity,
            priority or final civic action.
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = [
        ("01", "Evidence first", "Photo, description and location create the case record."),
        ("02", "Real AI analysis", "Supported road damage can be analysed with the trained detector."),
        ("03", "Clear uncertainty", "Unknown values stay unknown instead of being fabricated."),
        ("04", "Human verification", "Ambiguous or high-impact reports can stay pending review."),
        ("05", "City intelligence", "Reports can become useful area and category patterns."),
        ("06", "Outcome learning", "Verified outcomes can later support better models."),
    ]

    feature_cols = st.columns(3, gap="medium")

    for index, (number, title, copy) in enumerate(features):
        with feature_cols[index % 3]:
            st.markdown(
                f"""
                <div class="feature" style="margin-bottom:16px">
                    <div class="feature-no">{number}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-kicker">Workflow</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Report → analyse → verify → track</div>',
        unsafe_allow_html=True,
    )

    flow = [
        ("1", "Report", "Capture evidence."),
        ("2", "Analyse", "Run supported AI detection."),
        ("3", "Review", "Check what the system found."),
        ("4", "Submit", "Create a trackable case."),
        ("5", "Learn", "Use outcomes to improve the system."),
    ]

    flow_cols = st.columns(5, gap="medium")

    for col, (number, title, copy) in zip(flow_cols, flow):
        with col:
            st.markdown(
                f"""
                <div class="step">
                    <div class="step-no">{number}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------
# REPORT PROBLEM
# ---------------------------------------------------------------------
elif st.session_state.page == "Report Problem":
    st.markdown(
        '<div class="section-kicker">Citizen intake</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Report a civic problem</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="section-copy">
            Upload evidence, provide the location and description, then run
            AI analysis for supported road-damage cases.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.08, 0.92], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("1. Evidence")

            photo = st.file_uploader(
                "Upload a photo",
                type=["jpg", "jpeg", "png", "webp"],
                help="Use a clear photo where the civic issue is visible.",
            )

            if photo:
                st.session_state.draft_image_bytes = photo.getvalue()
                suffix = Path(photo.name).suffix.lower().replace(".", "")
                st.session_state.draft_image_suffix = suffix or "jpg"

                image = Image.open(io.BytesIO(st.session_state.draft_image_bytes)).convert(
                    "RGB"
                )
                st.image(
                    image,
                    width="stretch",
                    caption="Uploaded evidence",
                )

            description = st.text_area(
                "What is happening?",
                placeholder=(
                    "Example: Large pothole near the college gate; "
                    "vehicles are moving around it."
                ),
                height=145,
            )

    with right:
        with st.container(border=True):
            st.subheader("2. Report details")

            category = st.selectbox(
                "Problem category",
                ["Auto-detect"] + CATEGORIES,
                help="Auto-detect uses the road-damage model when an image is provided.",
            )

            city = st.text_input(
                "City",
                placeholder="Jaipur",
            )

            area = st.text_input(
                "Area / locality",
                placeholder="Malviya Nagar",
            )

            coords = st.text_input(
                "Coordinates (optional)",
                placeholder="26.8467, 75.8152",
                help="Use latitude, longitude.",
            )

            threshold = st.slider(
                "AI detection threshold",
                min_value=0.15,
                max_value=0.75,
                value=0.25,
                step=0.05,
                help="Higher values reduce low-confidence detections.",
            )

            st.markdown(
                """
                <div class="notice" style="margin-top:12px">
                    <div class="notice-title">Privacy reminder</div>
                    <div class="notice-copy">
                        Upload only information relevant to the civic issue.
                        Avoid identity documents and unnecessary personal data.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div style="height:18px"></div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "🔍  Analyse report",
        key="analyse_report",
        width="stretch",
    ):
        if not photo and not description.strip():
            st.warning("Please add a photo or a description before analysis.")
        else:
            st.session_state.analysis = None
            st.session_state.submitted_report_id = None

            use_road_model = (
                photo
                and category in {"Auto-detect", "Pothole / Road Damage"}
            )

            if use_road_model:
                image = Image.open(
                    io.BytesIO(st.session_state.draft_image_bytes)
                ).convert("RGB")

                with st.spinner("Analysing the image with NagarSetu..."):
                    detections, annotated, error = run_road_analysis(
                        image,
                        threshold,
                    )

                if error:
                    st.error(error)
                else:
                    st.session_state.analysis = {
                        "category": category,
                        "city": city,
                        "area": area,
                        "coords": coords,
                        "description": description,
                        "detections": detections or [],
                        "annotated": annotated,
                        "threshold": threshold,
                    }

            else:
                # For non-road categories, there is no trained detector in
                # this project yet. Keep the UI honest.
                st.session_state.analysis = {
                    "category": category,
                    "city": city,
                    "area": area,
                    "coords": coords,
                    "description": description,
                    "detections": [],
                    "annotated": None,
                    "threshold": threshold,
                }

    if st.session_state.analysis:
        analysis = st.session_state.analysis
        detections = analysis["detections"]

        st.markdown(
            '<div class="section-kicker">AI assessment</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-title">Review what NagarSetu found</div>',
            unsafe_allow_html=True,
        )

        summary = format_detection_summary(detections)

        if detections:
            top_confidence = max(
                detection["confidence"] for detection in detections
            )
            detected_count = len(detections)
            detected_labels = Counter(
                detection["label"] for detection in detections
            )
            dominant_label = detected_labels.most_common(1)[0][0]

            if analysis["category"] == "Auto-detect":
                final_category = "Pothole / Road Damage"
            else:
                final_category = analysis["category"]

            st.markdown(
                f"<div class='notice'>"
                f"<div class='notice-title'>AI result available</div>"
                f"<div class='notice-copy'>"
                f"The detector found {detected_count} supported road-damage region(s). "
                f"AI confidence reflects the detector's confidence in the visual class. "
                f"It is not a severity score, priority score or final civic decision."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            result_left, result_right = st.columns([1.05, 0.95], gap="large")

            with result_left:
                st.image(
                    analysis["annotated"],
                    width="stretch",
                    caption="AI-annotated image",
                )

            with result_right:
                with st.container(border=True):
                    st.markdown(
                        '<div class="result-label">Detected issue</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="result-title">{dominant_label}</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        '<span class="badge badge-green">AI detection available</span>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f"<div style='margin-top:18px'>"
                        f"<div class='info-row'><span class='info-key'>AI confidence</span>"
                        f"<span class='info-val'>{top_confidence:.1%}</span></div>"
                        f"<div class='info-row'><span class='info-key'>Detection count</span>"
                        f"<span class='info-val'>{detected_count}</span></div>"
                        f"<div class='info-row'><span class='info-key'>Detected types</span>"
                        f"<span class='info-val'>{summary}</span></div>"
                        f"<div class='info-row'><span class='info-key'>Severity</span>"
                        f"<span class='info-val'>Pending assessment</span></div>"
                        f"<div class='info-row'><span class='info-key'>Priority</span>"
                        f"<span class='info-val'>Pending verification</span></div>"
                        f"<div class='info-row'><span class='info-key'>Department</span>"
                        f"<span class='info-val'>Pending routing</span></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div style="height:12px"></div>',
                    unsafe_allow_html=True,
                )

                if st.button(
                    "✓  Submit this report",
                    key="submit_analysed",
                    width="stretch",
                ):
                    report_id = save_report(
                        user_id=str((st.session_state.user or {}).get("user_id", "")),
                        user_name=str((st.session_state.user or {}).get("name", "")),
                        user_email=str((st.session_state.user or {}).get("email", "")),
                        category=final_category,
                        description=analysis["description"],
                        city=analysis["city"],
                        area=analysis["area"],
                        coords=analysis["coords"],
                        image_bytes=st.session_state.draft_image_bytes,
                        image_suffix=st.session_state.draft_image_suffix,
                        ai_detections=summary,
                        ai_confidence=top_confidence,
                    )

                    st.session_state.submitted_report_id = report_id

        else:
            if analysis["category"] == "Auto-detect":
                category_label = "No supported road damage detected"
                helper_text = (
                    "The current road-damage model did not detect a supported "
                    "road-damage class above the selected threshold."
                )
            elif analysis["category"] == "Pothole / Road Damage":
                category_label = "No supported road damage detected"
                helper_text = (
                    "No supported road-damage class was detected above the "
                    "selected threshold."
                )
            else:
                category_label = analysis["category"]
                helper_text = (
                    "This category does not yet have a trained image detector "
                    "in the current NagarSetu model."
                )

            st.markdown(
                f"<div class='result'>"
                f"<div class='result-label'>Assessment</div>"
                f"<div class='result-title'>{category_label}</div>"
                f"<div class='section-copy'>{helper_text}</div>"
                f"<div class='info-row'><span class='info-key'>AI confidence</span>"
                f"<span class='info-val'>Not available</span></div>"
                f"<div class='info-row'><span class='info-key'>Severity</span>"
                f"<span class='info-val'>Pending assessment</span></div>"
                f"<div class='info-row'><span class='info-key'>Priority</span>"
                f"<span class='info-val'>Pending verification</span></div>"
                f"<div class='info-row'><span class='info-key'>Department</span>"
                f"<span class='info-val'>Pending routing</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div style="height:12px"></div>',
                unsafe_allow_html=True,
            )

            if st.button(
                "✓  Submit this report",
                key="submit_non_ai",
                width="stretch",
            ):
                report_id = save_report(
                        user_id=str((st.session_state.user or {}).get("user_id", "")),
                        user_name=str((st.session_state.user or {}).get("name", "")),
                        user_email=str((st.session_state.user or {}).get("email", "")),
                    category=analysis["category"]
                    if analysis["category"] != "Auto-detect"
                    else "Unclassified civic issue",
                    description=analysis["description"],
                    city=analysis["city"],
                    area=analysis["area"],
                    coords=analysis["coords"],
                    image_bytes=st.session_state.draft_image_bytes,
                    image_suffix=st.session_state.draft_image_suffix,
                )

                st.session_state.submitted_report_id = report_id

        if st.session_state.submitted_report_id:
            report_id = st.session_state.submitted_report_id

            st.success(
                f"Report submitted successfully. Your report ID is {report_id}."
            )

            st.markdown(
                f"""
                <div class="notice">
                    <div class="notice-title">Keep this report ID</div>
                    <div class="notice-copy">
                        {report_id} can be used to find the report in the
                        <b>My Reports</b> section.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------
# MY REPORTS
# ---------------------------------------------------------------------
elif st.session_state.page == "My Reports":
    st.markdown(
        '<div class="section-kicker">Case management</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">My reports</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="section-copy">
            Review submitted civic cases, AI evidence and current workflow state.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='notice'><div class='notice-title'>Your account records</div>"
        "<div class='notice-copy'>Only reports submitted from your signed-in account are shown here. "
        "Each new report is saved with your account ID and timestamp.</div></div>",
        unsafe_allow_html=True,
    )

    df = load_reports()

    current_user = st.session_state.user or {}
    current_user_id = str(current_user.get("user_id", ""))

    if current_user_id and "user_id" in df.columns:
        df = df[df["user_id"].astype(str) == current_user_id]

    if df.empty:
        st.info("You have not submitted any reports yet.")
    else:
        filter_left, filter_mid, filter_right = st.columns(3)

        with filter_left:
            status_values = ["All"] + sorted(
                df["status"].dropna().astype(str).unique().tolist()
            )
            status_filter = st.selectbox(
                "Status",
                status_values,
            )

        with filter_mid:
            category_values = ["All"] + sorted(
                df["category"].dropna().astype(str).unique().tolist()
            )
            category_filter = st.selectbox(
                "Category",
                category_values,
            )

        with filter_right:
            search = st.text_input(
                "Search",
                placeholder="Report ID, city, area...",
            )

        filtered = df.copy()

        if status_filter != "All":
            filtered = filtered[filtered["status"] == status_filter]

        if category_filter != "All":
            filtered = filtered[filtered["category"] == category_filter]

        if search.strip():
            query = search.strip().lower()
            mask = (
                filtered.astype(str)
                .apply(
                    lambda column: column.str.lower().str.contains(
                        query,
                        regex=False,
                        na=False,
                    )
                )
                .any(axis=1)
            )
            filtered = filtered[mask]

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.metric("Visible reports", len(filtered))
        with k2:
            st.metric(
                "Open",
                int((filtered["status"] == "Open").sum())
                if not filtered.empty
                else 0,
            )
        with k3:
            st.metric(
                "AI analysed",
                int(
                    filtered["ai_detections"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                    .sum()
                )
                if not filtered.empty
                else 0,
            )
        with k4:
            st.metric(
                "Resolved",
                int((filtered["status"] == "Resolved").sum())
                if not filtered.empty
                else 0,
            )

        display_columns = [
            "report_id",
            "created_at",
            "category",
            "city",
            "area",
            "ai_detections",
            "ai_confidence",
            "status",
            "department",
        ]

        st.dataframe(
            filtered[display_columns],
            width="stretch",
            hide_index=True,
        )

# ---------------------------------------------------------------------
# CITY INSIGHTS
# ---------------------------------------------------------------------
elif st.session_state.page == "City Insights":
    st.markdown(
        '<div class="section-kicker">Operations</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">City insights</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="section-copy">
            Use the reports that actually exist in the system to understand
            category volume and active cases. Advanced hotspot and forecasting
            models will be added only after suitable geographic and outcome data
            are available.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_reports()

    if df.empty:
        st.info("City insights will appear after the first reports are submitted.")
    else:
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            category_counts = (
                df["category"]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename("Reports")
            )

            st.markdown(
                '<div class="insight-card" style="padding:20px">',
                unsafe_allow_html=True,
            )
            st.subheader("Reports by category")
            st.bar_chart(category_counts)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            status_counts = (
                df["status"]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename("Reports")
            )

            st.markdown(
                '<div class="insight-card" style="padding:20px">',
                unsafe_allow_html=True,
            )
            st.subheader("Workflow status")
            st.bar_chart(status_counts)
            st.markdown("</div>", unsafe_allow_html=True)

        ai_count = int(
            df["ai_detections"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

        i1, i2, i3 = st.columns(3)

        with i1:
            st.metric("Total reports", len(df))

        with i2:
            st.metric("AI analysed reports", ai_count)

        with i3:
            st.metric(
                "Open cases",
                int((df["status"] == "Open").sum()),
            )

# ---------------------------------------------------------------------
# AI & TRUST
# ---------------------------------------------------------------------
elif st.session_state.page == "AI & Trust":
    st.markdown(
        '<div class="section-kicker">Responsible AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">AI & trust</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="section-copy">
            NagarSetu is designed to show what the model can actually support.
            Unknown values remain unknown until a validated model exists.
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_road_model()
    model_available = model is not None

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    trust_items = [
        (
            "Road detector",
            "Connected"
            if model_available
            else "Unavailable",
        ),
        ("Confidence", "Shown from model output"),
        ("Severity", "Not implemented"),
        ("Human review", "Supported by workflow"),
    ]

    for column, (title, value) in zip(
        [c1, c2, c3, c4],
        trust_items,
    ):
        with column:
            st.markdown(
                f"""
                <div class="feature">
                    <div class="feature-title">{title}</div>
                    <div class="feature-copy">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-kicker">Current model scope</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Supported road-damage classes</div>',
        unsafe_allow_html=True,
    )

    classes = [
        ("Pothole", "Supported"),
        ("Alligator Crack", "Supported"),
        ("Longitudinal Crack", "Supported"),
        ("Transverse Crack", "Supported"),
    ]

    class_cols = st.columns(4, gap="medium")

    for column, (name, status) in zip(class_cols, classes):
        with column:
            st.markdown(
                f"""
                <div class="card">
                    <div class="feature-title">{name}</div>
                    <div style="margin-top:10px">
                        <span class="badge badge-green">{status}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-kicker">Important distinction</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="notice">
            <div class="notice-title">Confidence is not severity</div>
            <div class="notice-copy">
                A high-confidence detection means the detector is confident
                about the visual class it predicted. It does not by itself
                prove how dangerous the road condition is, how urgent the
                repair is, or which department should own the case.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------
else:
    st.markdown(
        '<div class="section-kicker">Product</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">About NagarSetu</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <div class="result-title">Predictive Civic Problem Intelligence</div>
            <div class="section-copy">
                NagarSetu is a Python and Streamlit civic-tech project that
                turns citizen evidence into structured, traceable reports and
                uses machine learning where suitable.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-kicker">Product principles</div>',
        unsafe_allow_html=True,
    )

    principles = [
        ("Evidence", "Prefer observable evidence over assumptions."),
        ("Transparency", "Separate model confidence from civic decisions."),
        ("Verification", "Allow uncertain cases to remain pending review."),
        ("Traceability", "Keep report IDs and workflow state attached to cases."),
    ]

    pcols = st.columns(4, gap="medium")

    for column, (title, copy) in zip(pcols, principles):
        with column:
            st.markdown(
                f"""
                <div class="feature">
                    <div class="feature-title">{title}</div>
                    <div class="feature-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        NagarSetu • White / Blue / Yellow • Evidence-first civic intelligence
    </div>
    """,
    unsafe_allow_html=True,
)

