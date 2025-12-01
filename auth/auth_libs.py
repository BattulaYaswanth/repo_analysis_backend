from datetime import datetime,timedelta
import bcrypt,random,jwt,os
import smtplib
from email.mime.text import MIMEText
import asyncio

def hash_password(password:str):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'),salt)
    return hashed_password.decode('utf-8')

def verify_password(password:str,hashed_pw:str):
    if not bcrypt.checkpw(password.encode('utf-8'),hashed_password=hashed_pw.encode('utf-8')):
        return False
    else:
        return True
    
def create_jwt_token(user_id:str,email:str) -> str:
    secret_key = os.getenv("SECRET_KEY")
    if not secret_key:
        raise RuntimeError("SECRET_KEY not set in environment")
    payload = {
            "user_id": user_id,
            "email": email,
            "exp": datetime.now() + timedelta(seconds=int(os.getenv("ACCESS_TOKEN_EXPIRY")))# Expiration time
        }
    encoded_jwt = jwt.encode(payload, secret_key, algorithm="HS256")
    return encoded_jwt

def create_refresh_token(user_id:str,email:str) -> str:
    REFRESH_SECRET_KEY = os.getenv("SECRET_KEY")
    if not REFRESH_SECRET_KEY:
        raise RuntimeError("SECRET_KEY not set in environment")
    payload = {
            "user_id": user_id,
            "email": email,
            "type":"refresh",
            "exp": datetime.now() + timedelta(days=7)# Expiration time
        }
    encoded_refresh_token = jwt.encode(payload, REFRESH_SECRET_KEY, algorithm="HS256")
    return encoded_refresh_token

def is_jwt_token_valid(token:str) -> bool:
    secret_key = os.getenv("SECRET_KEY")
    if not secret_key:
        raise RuntimeError("SECRET_KEY not set in environment")
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        return payload.get("exp", 0) > datetime.now().timestamp()
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return False
    
def is_refresh_token_valid(token:str) -> bool:
    REFRESH_SECRET_KEY = os.getenv("SECRET_KEY")
    if not REFRESH_SECRET_KEY:
        raise RuntimeError("SECRET_KEY not set in environment")

    try:
        payload = jwt.decode(token, REFRESH_SECRET_KEY, algorithms=["HS256"])

        # Check expiration manually
        if payload.get("exp", 0) < datetime.now().timestamp():
            return {
                "valid": False,
                "error": "Refresh token expired"
            }

        # Return user data from payload
        return {
            "valid": True,
            "user_id": payload.get("user_id"),
            "email": payload.get("email")
        }

    except jwt.ExpiredSignatureError:
        return { "valid": False, "error": "Refresh token expired" }
    except jwt.InvalidTokenError:
        return { "valid": False, "error": "Invalid refresh token" }

    
def decode_jwt_token(token: str):
    secret_key = os.getenv("SECRET_KEY")
    try:
        decoded = jwt.decode(token, secret_key, algorithms=["HS256"])
        return decoded  
    except jwt.ExpiredSignatureError:
        raise Exception("Token expired")
    except jwt.InvalidTokenError:
        raise Exception("Invalid token")
    

def generate_otp() -> str:
    """Generates a 6-digit OTP."""
    return str(random.randint(100000, 999999))

def hash_otp(otp: str) -> str:
    """Hash an OTP using bcrypt for secure storage."""
    salt = bcrypt.gensalt()
    hashed_otp = bcrypt.hashpw(otp.encode('utf-8'), salt)
    return hashed_otp.decode('utf-8')

def verify_otp(provided_otp: str, hashed_otp: str) -> bool:
    """Verify a provided OTP against its hashed version."""
    return bcrypt.checkpw(provided_otp.encode('utf-8'), hashed_otp.encode('utf-8'))

OTP_MESSAGE_TEMPLATE = os.getenv("OTP_MESSAGE_TEMPLATE")
# Email configuration
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
async def send_otp_email(email: str, otp: str):
    """Send OTP email using configured SMTP server.

    - Uses SMTPS (`SMTP_SSL`) when `SMTP_PORT` is 465.
    - Uses STARTTLS (`SMTP(...); starttls()`) for other common ports (e.g., 587).
    - Runs the blocking SMTP calls in a thread via `asyncio.to_thread` so the event loop
      isn't blocked.
    """
    sender = SMTP_EMAIL
    password = SMTP_PASSWORD
    # Fallback message if template missing
    template = OTP_MESSAGE_TEMPLATE or "Your verification code is: {otp}"
    message = template.format(otp=otp)
    msg = MIMEText(message)
    msg["Subject"] = "Verify your account"
    msg["From"] = sender
    msg["To"] = email

    def _send():
        # Choose SSL vs STARTTLS based on port
        try:
            if SMTP_PORT == 465:
                with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                    server.login(sender, password)
                    server.sendmail(sender, [email], msg.as_string())
            else:
                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
                    server.ehlo()
                    # Start TLS for non-SSL port (common: 587)
                    try:
                        server.starttls()
                        server.ehlo()
                    except Exception:
                        # If starttls fails, continue and let login raise a meaningful error
                        pass
                    server.login(sender, password)
                    server.sendmail(sender, [email], msg.as_string())
        except Exception:
            # Re-raise to be handled by caller
            raise

    # Run the blocking send in a thread
    await asyncio.to_thread(_send)