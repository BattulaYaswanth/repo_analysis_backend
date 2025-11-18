from fastapi import FastAPI, APIRouter, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from bson.objectid import ObjectId
from github import Github
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from collections import defaultdict

import os, re, git, tempfile

# ─── Internal Imports ────────────────────────────────────────
from configurarions import collection, review_collection
from database.models import UserRegister, UserLogin, RepoInput,TokenRefresh
from database.schemas import all_users
from auth.auth_libs import (
    hash_password, verify_password,
    create_jwt_token, is_jwt_token_valid,
    decode_jwt_token,create_refresh_token,is_refresh_token_valid
)

# ─── FastAPI Setup ───────────────────────────────────────────
load_dotenv()
app = FastAPI(title="AI Developer Productivity API")
router = APIRouter()

# ─── CORS Setup ──────────────────────────────────────────────
origins = [
    "http://localhost:3000",
    "https://yourfrontenddomain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def init_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

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

@router.post("/api/auth/register")
async def register_user(user: UserRegister):
    try:
        if collection.find_one({"email": user.email}):
            raise HTTPException(status_code=400, detail="User already exists")
        if len(user.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

        user.password = hash_password(user.password)
        inserted = collection.insert_one(dict(user))
        token = create_jwt_token(str(inserted.inserted_id), user.email)

        # NOTE: This endpoint doesn't need to return the refresh token, only login does.
        return {"status": "success", "id": str(inserted.inserted_id), "access_token": token}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")

@router.post("/api/auth/login")
async def login_user(user: UserLogin): # Response removed as cookie is no longer set
    try:
        db_user = collection.find_one({"email": user.email})
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")
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
@router.post("/api/analyze_repo")
async def analyze_repo(data: RepoInput, authorization: str = Header(None)):
    token = extract_bearer_token(authorization)
    validate_token_or_401(token)
    repo_url = str(data.repo_url).removesuffix(".git")
    existing_repo = review_collection.find_one({"repo_url": repo_url})
    if existing_repo:
        return {
            "status": "success",
            "id":str(existing_repo["_id"]),
        }
    match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
    if match:
        owner = match.group(1)
        repo_name = match.group(2)
        print("Owner:", owner)
        print("Repo Name:", repo_name)
    else:
        print("Invalid GitHub URL")

    try:
        repo_path = clone_repo(data.repo_url)
        
        code_files = collect_code_files(repo_path)

        if not code_files:
            raise HTTPException(status_code=400, detail="No readable code files found")

        code_context = "\n\n".join(code_files)
        llm = init_llm()

        prompts = {
            "review": f"You are a senior engineer. Review this code:\n{code_context[:15000]}",
            "docs": f"You are a technical writer. Write a README for:\n{code_context[:15000]}",
            "tests": f"You are a QA engineer. Generate unit tests for:\n{code_context[:15000]}",
        }

        results = {key: llm.invoke([HumanMessage(content=prompt)]).content for key, prompt in prompts.items()}

        review_doc = {
            "owner": owner,
            "email": get_email(token),
            "repo_name": repo_name,
            "repo_url": repo_url,
            "languages": count_code_files(repo_path),
            "total_files": len(code_files),
            "has_readme": True, # Mock result
            "status": "completed",
            "review": results["review"],
            "docs": results["docs"],
            "tests": results["tests"],
        }
        # (Optional) Save results to MongoDB
        inserted = review_collection.insert_one(review_doc)

        return {"status": "success", "id": str(inserted.inserted_id)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing repo: {e}")

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
            "tests": repo.get("tests", "")
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

# ─── Include Router ─────────────────────────────────────────
app.include_router(router)