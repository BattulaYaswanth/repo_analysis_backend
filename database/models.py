from pydantic import BaseModel,EmailStr, Field,field_validator,HttpUrl
from datetime import datetime, timezone

def aware_utcnow():
    return datetime.now(timezone.utc)

class TokenRefresh(BaseModel):
    refresh_token: str

class UserBase(BaseModel):
    email: EmailStr = Field(..., example="user@example.com")

class UserRegister(UserBase):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=6, example="strongpassword123")
    created_at: datetime = Field(default_factory=aware_utcnow)
    updated_at: datetime = Field(default_factory=aware_utcnow)

class UserLogin(UserBase):
    password: str = Field(..., min_length=6)

# NEW Models for OTP
class OTPRequest(BaseModel):
    email: EmailStr

class OTPVerify(BaseModel):
    email: EmailStr
    otp: str

### GITHUB Urls Models ###

class RepoInput(BaseModel):
    """Input model for the repository URL."""
    repo_url: HttpUrl

    @field_validator('repo_url')
    def must_be_github(cls, value):
        if 'github.com' not in str(value):
            raise ValueError('URL must be a valid GitHub repository link.')
        return value

class RepoDetails(BaseModel):
    """Model to return the analysis results to the frontend."""
    owner: str
    email:str
    repo_name: str
    repo_url:str
    used_model:str
    languages: dict[str, float]
    total_files: int
    has_readme: bool
    status: str
    review:str
    docs:str
    tests:str

class CodeInput(BaseModel):
    code:str
    language:str
