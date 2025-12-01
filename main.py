from fastapi import FastAPI, APIRouter, HTTPException, Header , BackgroundTasks,Depends
from fastapi.responses import StreamingResponse,PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from bson.objectid import ObjectId
from github import Github
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from collections import defaultdict
from cachetools import TTLCache
from typing import AsyncGenerator
from datetime import timedelta, timezone
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter

import os, re, git, tempfile,datetime,random
import google.generativeai as genai
# ─── Internal Imports ────────────────────────────────────────
from configurations import collection, review_collection
from database.models import UserRegister, UserLogin, RepoInput,TokenRefresh,CodeInput
from database.schemas import all_users
from auth.auth_libs import (
    hash_password, verify_password,
    create_jwt_token, is_jwt_token_valid,
    decode_jwt_token,create_refresh_token,is_refresh_token_valid,
    generate_otp,send_otp_email,hash_otp,verify_otp as verify_otp_hash
)
from tokens.check_tokens import get_remaining_tokens

# ─── FastAPI Setup ───────────────────────────────────────────
load_dotenv()
# REMOVE: dependencies=[Depends(RateLimiter(times=10, seconds=60))] from app declaration
# This dependency should only be on specific routes or the APIRouter.
app = FastAPI(title="AI Developer Productivity API")
router = APIRouter()
# 1. Get the string from env
raw_origins = os.getenv("ALLOWED_ORIGINS", "")
# 2. Parse it into a list

# ─── CORS Setup ──────────────────────────────────────────────
if raw_origins:
    origins = [origin.strip() for origin in raw_origins.split(",")]
else:
    # Fallback if env var is missing (optional)
    origins = ["http://localhost:3000"]
print(f"🌍 Allowed CORS Origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── RATE LIMITER INITIALIZATION (REQUIRES REDIS) ─────────────

@app.on_event("startup")
async def startup():
    """Initialize Redis (Redis Labs) and FastAPILimiter on startup."""
    
    # --- 1. Configuration for Redis Labs ---
    # Your Redis Labs hostname and port
    REDIS_HOST = os.getenv("REDIS_URL")
    REDIS_PORT = os.getenv("REDIS_PORT")
    
    # Retrieve the password from the environment variables (CRITICAL)
    # Ensure REDIS_PASSWORD is set in your .env or environment
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") 
    
    if not REDIS_PASSWORD:
        print("❌ CRITICAL: REDIS_PASSWORD environment variable is not set. Rate limiting will not be initialized.")
        return 
        
    print(f"🔗 Attempting secure connection to Redis Labs at: {REDIS_HOST}:{REDIS_PORT}")

    try:
        # --- 2. Create the Secure Connection Client ---
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            username="default",
            decode_responses=True
        )
        
        # --- 3. Test Connection and Initialize Limiter ---
        await redis_client.ping() 
        print("✅ Secure Redis connection successful.")
        await FastAPILimiter.init(redis_client)
        print("✅ FastAPILimiter initialized.")
        
    except Exception as e:
        print(f"❌ Failed to connect to Redis Labs/initialize FastAPILimiter: {e}")
        # Optionally, raise the exception here if Redis is a hard dependency
        # raise Exception("Failed to connect to Redis for rate limiting.") from e

# ─── GitHub Setup ────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
github_client = Github(GITHUB_TOKEN) if GITHUB_TOKEN else Github()
if not GITHUB_TOKEN:
    print("⚠️ WARNING: GITHUB_TOKEN not found. Only public repos can be accessed.")

# ─── Helper Functions ────────────────────────────────────────
# (Keeping original helper functions)
def extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0] != "Bearer":
        raise HTTPException(status_code=401, detail="Invalid token format. Expected 'Bearer <token>'")
    return parts[1]

def validate_token_or_401(token: str):
    if not is_jwt_token_valid(token):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
def validate_refresh_token_or_401(token: str):
    if not token:
            raise HTTPException(status_code=400, detail="Refresh token is required")
    token_data = is_refresh_token_valid(token)
    if not token_data["valid"]:
        raise HTTPException(status_code=401, detail=token_data["error"])
    return token_data

def is_valid_email(email: str) -> bool:
    regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(regex, email) is not None

# --------------------------------------
# Clone Repository
# --------------------------------------
def clone_repo(repo_url: str) -> str:
    tmp_dir = tempfile.mkdtemp()
    git.Repo.clone_from(repo_url, tmp_dir)
    return tmp_dir


# --------------------------------------
# List of Code Extensions
# --------------------------------------
CODE_EXTENSIONS = (
    ".py", ".ipynb",                        # Python / Jupyter
    ".js", ".jsx", ".ts", ".tsx",           # JavaScript / TypeScript
    ".java", ".kt", ".kts",                 # Java / Kotlin
    ".go", ".rs",                           # Go / Rust
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp",     # C / C++
    ".rb", ".php", ".swift",                # Ruby / PHP / Swift
    ".cs",                                  # C#
    ".scala", ".clj", ".r",                 # Scala / Clojure / R
    ".html", ".htm", ".css", ".scss", ".sass", ".vue", ".svelte",  # Web
    ".sql", ".prisma", ".graphql", ".gql",  # Database / Schema
    ".json", ".yaml", ".yml", ".toml", ".ini", ".env",  # Config
    ".sh", ".bash", ".zsh", ".ps1", ".bat",  # Scripts
    ".md", ".txt", ".rst"                   # Docs
)


# --------------------------------------
# Collect Code File Contents
# --------------------------------------
def collect_code_files(repo_path: str, extensions=CODE_EXTENSIONS) -> list[str]:
    code_files = []

    for root, _, files in os.walk(repo_path):
        for f in files:
            if f.lower().endswith(extensions):
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore") as file:
                        code_files.append(file.read())
                except Exception:
                    pass
    return code_files


# --------------------------------------
# Count Code Files By Extension
# --------------------------------------
def count_code_files(repo_path: str, extensions=CODE_EXTENSIONS) -> dict[str, int]:
    counts = defaultdict(int)

    for root, _, files in os.walk(repo_path):
        for f in files:
            for ext in extensions:
                if f.lower().endswith(ext.lower()):
                    counts[ext.lstrip('.')] += 1
                    break  # Avoid double-counting

    return dict(counts)


def get_email(token:str) -> str:
    decoded = decode_jwt_token(token)
    return decoded["email"]

# ─────────────────────────────────────────────────────────────
# LLM Initialization Helper
# ─────────────────────────────────────────────────────────────

def get_available_models():
    """Parses the env var and returns a list of model names."""
    models = os.getenv("GROQ_MODELS", "")
    return [m.strip() for m in models.split(",") if m.strip()]

def get_llm_client(model_name: str):
    """Instantiates the ChatGroq client for a specific model."""
    return ChatGroq(
        model=model_name,
        temperature=1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

# ─────────────────────────────────────────────────────────────
# BACKGROUND ANALYSIS TASK (With Rate Limit Fallback)
# ─────────────────────────────────────────────────────────────

def run_repo_analysis_task(repo_url: str, review_id: str, token: str, owner: str, repo_name: str):
    try:
        set_progress(review_id, 5)

        # 1. Clone and Prep
        repo_path = clone_repo(repo_url)
        set_progress(review_id, 20)

        code_files = collect_code_files(repo_path)
        if not code_files:
            review_collection.delete_one(
                {"_id": ObjectId(review_id)}
            )
            set_progress(review_id, 100)
            return

        # 2. Prepare Context
        # Note: We cap at 15k chars, but this might still result in high token counts
        code_context = "\n\n".join(code_files)[:15000] 
        
        # Create LangChain Message Object
        messages = [HumanMessage(content=f"Analyze:\n{code_context}")]

        # 3. Get models and shuffle
        model_candidates = get_available_models()
        random.shuffle(model_candidates)
        
        print(f"🔍 Starting Analysis on {repo_name}...")
        print(f"📂 Candidates: {model_candidates}")

        final_results = None
        used_model = None

        # 4. THE FALLBACK LOOP
        # We try each model. If one works, we break. If one hits a rate limit, we continue.
        for model in model_candidates:
            try:
                print(f"👉 Trying model: {model}")

                # A. Check Context Window (Memory)
                # Using the updated check_tokens logic you implemented
                token_data = get_remaining_tokens(model, messages)
                remaining = token_data.get("remaining_tokens", 0)

                if remaining < 2000:
                    print(f"   Skipping {model}: Context full (Remaining: {remaining})")
                    continue

                # B. Initialize Client
                current_llm = get_llm_client(model)

                # C. Attempt Generation (This is where 413/429 happens)
                # We run the prompts here to verify the model accepts the load
                prompts_to_run = {
                    "review": f"You are a senior engineer. Review this code:\n{code_context}",
                    "docs": f"Write a high-quality README for this repository:\n{code_context}",
                    "tests": f"Generate unit tests for this code:\n{code_context}",
                }
                
                results = {}
                total_steps = len(prompts_to_run)
                
                for i, (key, prompt_text) in enumerate(prompts_to_run.items()):
                    msg = [HumanMessage(content=prompt_text)]
                    # This line will throw the 413 error if TPM is exceeded
                    response = current_llm.invoke(msg)
                    results[key] = response.content
                    
                    # Update progress relative to the attempt
                    set_progress(review_id, 40 + int((i + 1) * 20))

                # D. If we get here, SUCCESS!
                final_results = results
                used_model = model
                print(f"✅ Success using model: {model}")
                break 

            except Exception as e:
                error_str = str(e).lower()
                # Check for Rate Limit (429) or Size Limit (413) errors
                if "rate_limit" in error_str or "413" in error_str or "429" in error_str:
                    print(f"   ⚠️ Rate Limit hit on {model}. Switching to next model...")
                    continue # Try the next model in the list
                else:
                    # If it's a logic error (e.g., code bug), raise it
                    print(f"   ❌ Unexpected error on {model}: {e}")
                    raise e

        # 5. Check if we found a working model
        if not final_results:
            raise Exception("All models failed due to Rate Limits or Context limits.")

        # 6. Save to DB
        review_collection.update_one(
            {"_id": ObjectId(review_id)},
            {"$set": {
                "owner": owner,
                "email": get_email(token),
                "repo_name": repo_name,
                "languages": count_code_files(repo_path),
                "total_files": len(code_files),
                "status": "completed",
                "review": final_results["review"],
                "docs": final_results["docs"],
                "tests": final_results["tests"],
                "used_model": used_model 
            }}
        )
        set_progress(review_id, 100)

    except Exception as e:
        print(f"❌ FATAL Job Failed: {e}")
        review_collection.delete_one(
            {"_id": ObjectId(review_id)}
        )
        set_progress(review_id, 100)

def set_progress(job_id: str, value: int):
    JOB_PROGRESS[job_id] = min(max(value, 0), 100)


# Global Progress Store (non-blocking, works per-job)
JOB_PROGRESS = defaultdict(lambda: 0)

# ─── Routes: Auth ────────────────────────────────────────────
@router.get("/api/auth/validate-token")
async def validate_token(authorization: str = Header(None)):
    token = extract_bearer_token(authorization)
    validate_token_or_401(token)
    return {"status": "valid"}

@router.get("/api/auth")
async def get_users():
    users = collection.find()
    return all_users(users)

@router.get("/api/auth/reviews")
async def get_reviews(email:str,authorization: str = Header(None)):
    try:
        token = extract_bearer_token(authorization)
        validate_token_or_401(token)
        
        db_user = collection.find_one({"email": email})
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Convert synchronous find to list
        reviews = list(review_collection.find({"email": email}))
        
        # Convert ObjectId to string for JSON serialization
        for review in reviews:
            review["_id"] = str(review["_id"])
        
        return {"status": "success", "reviews": reviews}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")
    
@router.post("/api/user/verify-otp")
async def verify_otp_endpoint(data: dict):
    try:
        # Lookup user
        user = collection.find_one({"email": data.get("email")})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Already verified
        if user.get("verified"):
            return {"status": "success", "message": "User already verified"}

        # Validate provided OTP
        provided_otp_raw = data.get("otp")
        if provided_otp_raw is None:
            raise HTTPException(status_code=400, detail="OTP is required")
        provided_otp = str(provided_otp_raw).strip()
        if not provided_otp.isdigit():
            raise HTTPException(status_code=400, detail="Invalid OTP format")

        # Verify hashed OTP
        hashed_otp = user.get("otp")
        if not hashed_otp:
            raise HTTPException(status_code=400, detail="Invalid OTP")
        
        if not verify_otp_hash(provided_otp, hashed_otp):
            raise HTTPException(status_code=400, detail="Invalid OTP")

        # Expiry check (coerce DB value to timezone-aware UTC if needed)
        expiry = user.get("otp_expiry")
        if not expiry:
            raise HTTPException(status_code=400, detail="OTP expired")

        # PyMongo often returns naive datetimes (UTC without tzinfo). Coerce to
        # timezone-aware UTC so comparisons with `datetime.now(timezone.utc)` work.
        try:
            if isinstance(expiry, datetime.datetime) and expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
        except Exception:
            # If expiry isn't a datetime for some reason, fail as expired
            raise HTTPException(status_code=400, detail="OTP expired")

        if datetime.datetime.now(timezone.utc) > expiry:
            raise HTTPException(status_code=400, detail="OTP expired")

        # Mark user as verified and clear OTP fields
        collection.update_one(
            {"email": user["email"]},
            {"$set": {"verified": True}, "$unset": {"otp": "", "otp_expiry": "","last_otp_sent":""}}
        )

        return {"status": "success", "message": "Account verified successfully"}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")
    
@router.post("/api/user/resend-otp")
async def resend_otp(data: dict):
    try:
        user = collection.find_one({"email": data.get("email")})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if user.get("verified"):
            return {"status": "success", "message": "User already verified"}

        # Rate limiting: 60-second cooldown between resend attempts
        last_otp_sent = user.get("last_otp_sent")
        if last_otp_sent:
            if isinstance(last_otp_sent, datetime.datetime) and last_otp_sent.tzinfo is None:
                last_otp_sent = last_otp_sent.replace(tzinfo=timezone.utc)
            
            time_elapsed = (datetime.datetime.now(timezone.utc) - last_otp_sent).total_seconds()
            if time_elapsed < 60:
                seconds_remaining = int(60 - time_elapsed)
                raise HTTPException(
                    status_code=429,
                    detail=f"Please wait {seconds_remaining} seconds before requesting a new OTP"
                )

        otp = generate_otp()
        hashed_otp = hash_otp(otp)
        otp_expiry = datetime.datetime.now(timezone.utc) + timedelta(minutes=10)
        otp_ttl_seconds = 600  # 10 minutes
        
        collection.update_one(
            {"email": user["email"]},
            {"$set": {
                "otp": hashed_otp,
                "otp_expiry": otp_expiry,
                "last_otp_sent": datetime.datetime.now(timezone.utc)
            }}
        )

        # Send OTP email
        await send_otp_email(user["email"], f"Your new OTP is: {otp}")
        
        return {
            "status": "success",
            "message": "OTP resent successfully",
            "otp_ttl_seconds": otp_ttl_seconds
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")



@router.post("/api/auth/register",
             dependencies=[Depends(RateLimiter(times=20, seconds=60))])
async def register_user(user: UserRegister):
    try:
        if not is_valid_email(user.email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        existing_user = collection.find_one({"email": user.email})

        # If user already exists and is verified, reject registration
        if existing_user and existing_user.get("verified"):
            raise HTTPException(status_code=400, detail="User already exists")

        # If user exists but is not verified, re-issue OTP and update password/hash if provided
        otp = generate_otp()
        hashed_otp = hash_otp(otp)
        otp_expiry = datetime.datetime.now(timezone.utc) + timedelta(minutes=10)
        otp_ttl_seconds = 600  # 10 minutes

        if existing_user and not existing_user.get("verified"):
            # Optionally update password if a new one provided
            updated_fields = {}
            if user.password:
                updated_fields["password"] = hash_password(user.password)
            updated_fields.update({
                "otp": hashed_otp,
                "otp_expiry": otp_expiry,
                "last_otp_sent": datetime.datetime.now(timezone.utc)
            })
            collection.update_one({"email": user.email}, {"$set": updated_fields})
            await send_otp_email(user.email, f"Your OTP verification code is: {otp}")
            return {
                "status": "pending_verification",
                "id": str(existing_user.get("_id")),
                "message": "OTP re-sent to email. Verify to activate your account.",
                "otp_ttl_seconds": otp_ttl_seconds
            }

        if len(user.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

        # Create new user (unverified) and send OTP
        user_doc = dict(user)
        user_doc["password"] = hash_password(user.password)
        user_doc["verified"] = False
        user_doc["otp"] = hashed_otp
        user_doc["otp_expiry"] = otp_expiry
        user_doc["last_otp_sent"] = datetime.datetime.now(timezone.utc)

        inserted = collection.insert_one(user_doc)
        await send_otp_email(user.email, f"Your OTP verification code is: {otp}")

        return {
            "status": "pending_verification",
            "id": str(inserted.inserted_id),
            "message": "OTP sent to email. Verify to activate your account.",
            "otp_ttl_seconds": otp_ttl_seconds
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")


@router.post("/api/auth/login",dependencies=[Depends(RateLimiter(times=15, seconds=60))])
async def login_user(user: UserLogin):
    try:
        db_user = collection.find_one({"email": user.email})
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")
        if not db_user["verified"]:
            raise HTTPException(status_code=403, detail="User not verified")
        if not verify_password(user.password, db_user["password"]):
            raise HTTPException(status_code=400, detail="Invalid credentials")

        # 1. Generate Access and Refresh Tokens
        access_token = create_jwt_token(str(db_user["_id"]), db_user["email"])
        refresh_token = create_refresh_token(str(db_user["_id"]), db_user["email"])
        
        # 2. Return tokens and expiry in JSON body (Required by NextAuth's custom flow)
        expiry = os.getenv("ACCESS_TOKEN_EXPIRY")
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": expiry # 1 hour, standard TTL for the access token
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")
    
@router.post("/api/auth/refresh")
async def refresh_access_token(data: TokenRefresh):
    """
    Refreshes the access token using a valid refresh token.
    """
    try:
        print("🔁 Incoming Refresh Token:", data.refresh_token)

        # 1. Validate refresh JWT
        try:
            payload = is_refresh_token_valid(data.refresh_token)
            print("🔓 Refresh Token Payload:", payload)
        except ValueError as e:
            raise HTTPException(status_code=401, detail=f"Invalid refresh token: {e}")
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

        user_id = payload.get("user_id")
        email = payload.get("email")

        if not user_id or not email:
            raise HTTPException(status_code=401, detail="Malformed token payload")

        # 2. Create a new access token
        new_access_token = create_jwt_token(user_id, email)
        expiry = os.getenv("ACCESS_TOKEN_EXPIRY", "3600")

        # 3. Send the new tokens back
        return {
            "access_token": new_access_token,
            "refresh_token": data.refresh_token,  # keeping same refresh token
            "expires_in": expiry,
        }

    except HTTPException:
        # Already handled errors → rethrow
        raise
    except Exception as e:
        print("🔥 Server Error:", e)
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

@router.delete("/api/auth/delete/{user_id}")
async def delete_user(user_id: str):
    try:
        user_obj_id = ObjectId(user_id)
        if not collection.find_one({"_id": user_obj_id}):
            raise HTTPException(status_code=404, detail="User not found")

        collection.delete_one({"_id": user_obj_id})
        return {"message": "User deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")

# ─── Routes: Repo Analysis ──────────────────────────────────
# (No changes needed in analysis routes)
@router.post("/api/analyze_repo", status_code=202,
            dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def analyze_repo(data: RepoInput, background_tasks: BackgroundTasks, authorization: str = Header(None)):

    token = extract_bearer_token(authorization)
    validate_token_or_401(token)

    repo_url = str(data.repo_url).removesuffix(".git")

    match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
    owner = match.group(1) if match else "unknown"
    repo_name = match.group(2) if match else repo_url.split("/")[-1]

    # If exists and finished → return cached
    existing = review_collection.find_one({"repo_url": repo_url, "status": "completed"})
    if existing:
        return {"status": "success", "id": str(existing["_id"])}

    new_doc = {
        "repo_url": repo_url,
        "email": get_email(token),
        "status": "pending",
        "created_at": datetime.datetime.now(timezone.utc),
    }

    inserted = review_collection.insert_one(new_doc)
    job_id = str(inserted.inserted_id)

    background_tasks.add_task(run_repo_analysis_task, repo_url, job_id, token, owner, repo_name)

    return {"status": "job_submitted", "id": job_id}

@router.get("/api/analyze_repo/{id}")
def get_review_by_id(id: str, authorization: str = Header(None)):
    try:
        token = extract_bearer_token(authorization)
        validate_token_or_401(token)

        if not ObjectId.is_valid(id):
            raise HTTPException(status_code=400, detail="Invalid Id format")

        repo = review_collection.find_one({"_id": ObjectId(id)})
        if not repo:
            raise HTTPException(status_code=404, detail="Wrong Id Provided")

        return {
            "status": "success",
            "id": str(repo["_id"]),
            "repo_name": repo.get("repo_name", ""),
            "repo_url": repo.get("repo_url", ""),
            "total_files": repo.get("total_files", 0),
            "languages": repo.get("languages", {}),
            "review": repo.get("review", ""),
            "docs": repo.get("docs", ""),
            "tests": repo.get("tests", ""),
            "used_model":repo.get("used_model","")
        }

    except HTTPException:
        raise  # re-throw normal errors

    except Exception as e:
        # unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")
    
@app.post("/api/auth/logout")
def logout():
    # Logout is simplified since we are not relying on backend cookies anymore.
    # The client-side (NextAuth) handles the session termination.
    return {"message": "Logged out"}

# Re-use your existing get_review_by_id route and adapt it:

STATUS_CACHE = TTLCache(maxsize=1000, ttl=60)
@router.get("/api/analyze_repo/status/{id}")
def get_review_status_and_result(id: str, authorization: str = Header(None)):
    try:
        # Authentication and ID validation (Keep as is)
        token = extract_bearer_token(authorization)
        validate_token_or_401(token)
        if not ObjectId.is_valid(id):
            raise HTTPException(status_code=400, detail="Invalid Id format")
        
        # --- CACHE CHECK ---
        cached_status = STATUS_CACHE.get(id)
        if cached_status:
            # If still running, return the cached status immediately
            if cached_status in ["pending", "processing"]:
                return {
                    "status": cached_status, 
                    "id": id, 
                    "message": f"Analysis is currently {cached_status}. (Cached)"
                }
            # If cached_status is "completed" or "failed", we still need the full payload from DB, 
            # so we skip the cache hit and proceed to DB.

        # --- DATABASE LOOKUP (Cache Miss) ---
        repo = review_collection.find_one({"_id": ObjectId(id)})
        if not repo:
            raise HTTPException(status_code=404, detail="Wrong Id Provided")

        current_status = repo.get("status", "unknown")
        
        # --- CACHE UPDATE LOGIC ---
        if current_status in ["pending", "processing"]:
            # Update cache with the status read from the database
            STATUS_CACHE[id] = current_status
            
            # Return the current status (from DB read)
            return {
                "status": current_status,
                "id": str(repo["_id"]),
                "message": f"Analysis is currently {current_status}."
            }

        # If we reach here, the job is either 'completed', 'failed', or 'unknown'
        
        # 1. If not completed (i.e., failed or unknown), return the status and error if present
        if current_status != "completed":
            # Optional: Cache 'failed' status as well, but generally less critical than running status
            # STATUS_CACHE[id] = current_status 
            
            return {
                "status": current_status,
                "id": str(repo["_id"]),
                "error": repo.get("error", "An error occurred."), # Include error field for failed status
                "message": f"Analysis is currently {current_status}."
            }

        # 2. If completed, return the full result
        # We do NOT cache the large, final result payload here to save cache memory.
        return {
            "status": "completed", # Change to "completed" instead of "success" for consistency
            "id": str(repo["_id"]),
            "repo_name": repo.get("repo_name", ""),
            "repo_url": repo.get("repo_url", ""),
            "used_model":repo.get("used_model",""),
            "total_files": repo.get("total_files", 0),
            "languages": repo.get("languages", {}),
            "review": repo.get("review", ""),
            "docs": repo.get("docs", ""),
            "tests": repo.get("tests", "")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")
    
@router.get("/api/analyze_repo/progress/{id}")
def get_progress(id: str, authorization: str = Header(None)):
    token = extract_bearer_token(authorization)
    validate_token_or_401(token)

    if not ObjectId.is_valid(id):
        raise HTTPException(status_code=400, detail="Invalid Id format")

    progress = JOB_PROGRESS.get(id, 0)

    return { "id": id, "progress": progress }

PROMPT_TEMPLATE = """
**ROLE:** You are an expert Senior Software Engineer and Code Quality Analyst.

**TASK:** Analyze the provided code snippet for common errors, logical flaws, performance bottlenecks, and potential security vulnerabilities.

1.  Structure your findings clearly under a main heading: **## Code Review Report**.
2.  For each issue, use a sub-heading (e.g., **### Issue 1: Missing Type Hints**) followed by:
    * **Description:** A clear explanation of the problem and its impact.
    * **Suggested Fix:** A concrete, corrected code snippet or instruction for correction.
3.  If the code is flawless, output only: **## Code Review Report\n\nCode appears clean and ready for deployment. 👍**
4.  The analysis language is **{language}**.

**THE CODE TO REVIEW:**
---
{code_input}
---
"""

# Cache: key = code text, value = full generated review text
CODEREVIEW_CACHE = TTLCache(maxsize=1000, ttl=1800)  # TTL = 30 Minutes

@router.post("/api/code_review")
async def code_review(request: CodeInput, authorization: str = Header(None)):

    try:
        # -------------------------------
        # 1. AUTH
        # -------------------------------
        token = extract_bearer_token(authorization)
        validate_token_or_401(token)

        # -------------------------------
        # 2. CACHE CHECK
        # -------------------------------
        cache_key = f"{request.language}-{hash(request.code)}"

        cached_output = CODEREVIEW_CACHE.get(cache_key)
        if cached_output:
            print("🚀 Cache HIT")
            # Return EXACTLY the markdown, not JSON
            return PlainTextResponse(
                cached_output,
                media_type="text/markdown"
            )


        print("🚀 Cache MISS — calling Gemini API")

        # -------------------------------
        # 3. Configure Gemini
        # -------------------------------
        API_KEY = os.getenv("GEMINI_API_KEY")
        if not API_KEY:
            raise ValueError("No API Key found. Set GEMINI_API_KEY in .env")

        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-2.5-pro')

        final_prompt = PROMPT_TEMPLATE.format(
            language=request.language,
            code_input=request.code
        )

        # -------------------------------
        # 4. Streaming Generator (also saves to cache)
        # -------------------------------
        async def review_generator() -> AsyncGenerator[str, None]:
            full_output = ""  # collect full text to store in cache

            try:
                response_stream = await model.generate_content_async(
                    final_prompt,
                    stream=True
                )

                async for chunk in response_stream:
                    if chunk.text:
                        full_output += chunk.text
                        yield chunk.text

                # Save final output to cache
                CODEREVIEW_CACHE[cache_key] = full_output
                print("💾 Saved response to cache.")

            except Exception as e:
                error_message = f"\n\n--- ERROR ---\nGemini API Error: {str(e)}"
                yield error_message

        # Return streaming response
        return StreamingResponse(review_generator(), media_type="text/markdown")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# ─── Include Router ─────────────────────────────────────────
app.include_router(router)