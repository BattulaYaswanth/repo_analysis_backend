import bcrypt,random,jwt,os
import resend
import secrets
from datetime import datetime,timedelta
from resend import Emails
from configurations import collection
from bson.objectid import ObjectId

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

def is_jwt_token_valid_and_active(token: str) -> bool:
    secret_key = os.getenv("SECRET_KEY")
    if not secret_key:
        raise RuntimeError("SECRET_KEY not set in environment")
    
    try:
        # 1. Decode the token (This checks signature and expiration)
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        
        user_id = payload.get("user_id") # Assuming user ID is stored here
        if not user_id:
            return False # Token is structurally invalid

        # 2. STATEFUL CHECK: Look up the user in the database
        db_user = collection.find_one({"_id": ObjectId(user_id)}) # Use the correct object ID type
        
        if not db_user:
            return False 

        # 3. If the user exists and the token hasn't expired, it's valid
        return True 

    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, jwt.DecodeError):
        # Catches expired, invalid signature, or malformed tokens
        return False
    except Exception as e:
        # Handle database errors or other unexpected issues
        print(f"Validation error: {e}")
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
    

def generate_otp(length=6):
    return ''.join(str(secrets.randbelow(10)) for _ in range(length))


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
async def send_otp_email(email: str, otp: str):
    resend.api_key = os.getenv("RESEND_API")
    template = OTP_MESSAGE_TEMPLATE or "Your verification code is: {otp}"
    message = template.format(otp=otp)
    payload = {
        "from": "onboarding@resend.dev",
        "to": email,
        "subject": "Verification Code",
        "html": f"<p><b>{message}</b></p>",
    }
    return Emails.send(payload)