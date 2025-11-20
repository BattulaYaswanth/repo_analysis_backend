from datetime import datetime
def individual_user(user):
    return{
        "id":str(user["_id"]),
        "email":str(user["email"]),
        "username":str(user["username"]),
        "created_at":datetime.fromisoformat(str(user["created_at"]))
    }

def all_users(users):
    return [individual_user(user) for user in users]

def individual_review(review) -> dict:
    """Convert MongoDB review document into a serializable dict."""
    return {
        "owner": str(review.get("owner", "")),
        "repo_name": str(review.get("repo_name", "")),
        "repo_url": str(review.get("repo_url", "")),
        "used_model":str(review.get("used_model","")),
        "languages": dict(review.get("languages", {})),
        "total_files": int(review.get("total_files", 0)),
        "has_readme": bool(review.get("has_readme", False)),
        "status": str(review.get("status", "Pending")),
        "review": str(review.get("review", "")),
        "docs": str(review.get("docs", "")),
        "tests": str(review.get("tests", "")),
    }

def all_reviews(reviews):
    return [individual_review(review) for review in reviews]