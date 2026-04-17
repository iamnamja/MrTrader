"""
Email alert sender using Python's stdlib smtplib.

Supports two modes:
  - Gmail SMTP (SMTP_HOST=smtp.gmail.com, port 587, TLS)
  - Any generic SMTP relay

Required env vars (all optional — email silently disabled if absent):
  SMTP_HOST       e.g. smtp.gmail.com
  SMTP_PORT       default 587
  SMTP_USER       sender address / login
  SMTP_PASSWORD   app password or SMTP password
  ALERT_EMAIL     recipient address (already in config.py)
"""
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


def send_email_alert(
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
) -> bool:
    """
    Send an email alert. Returns True if sent, False if skipped/failed.
    All SMTP config is read lazily so import never fails.
    """
    smtp_host = getattr(settings, "smtp_host", None)
    smtp_port = int(getattr(settings, "smtp_port", 587))
    smtp_user = getattr(settings, "smtp_user", None)
    smtp_password = getattr(settings, "smtp_password", None)
    recipient = getattr(settings, "alert_email", None)

    if not all([smtp_host, smtp_user, smtp_password, recipient]):
        logger.debug("Email alert skipped: SMTP not fully configured")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[MrTrader] {subject}"
    msg["From"] = smtp_user  # type: ignore[arg-type]
    msg["To"] = recipient  # type: ignore[arg-type]
    msg.attach(MIMEText(body_text, "plain"))
    if body_html:
        msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:  # type: ignore[arg-type]
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_password)  # type: ignore[arg-type]
            server.sendmail(smtp_user, recipient, msg.as_string())  # type: ignore[arg-type]
        logger.info("Email alert sent: %s → %s", subject, recipient)
        return True
    except Exception as exc:
        logger.warning("Email alert failed: %s", exc)
        return False


def build_alert_html(severity: str, title: str, message: str) -> str:
    color = {"CRITICAL": "#ef4444", "WARNING": "#eab308", "INFO": "#22c55e"}.get(severity, "#38bdf8")
    return f"""
<html><body style="font-family:monospace;background:#0b0e14;color:#d1d5db;padding:24px">
  <div style="max-width:480px;margin:0 auto">
    <h2 style="color:{color};margin-bottom:8px">{title}</h2>
    <p style="color:#6b7280;font-size:12px;margin-bottom:16px">Severity: <strong style="color:{color}">{severity}</strong></p>
    <div style="background:#131720;border:1px solid #1e2d40;border-left:3px solid {color};
                border-radius:6px;padding:16px;font-size:13px">{message}</div>
    <p style="color:#6b7280;font-size:11px;margin-top:16px">— MrTrader Automated Alert</p>
  </div>
</body></html>
"""
