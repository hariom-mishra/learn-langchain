from pydantic import BaseModel
from typing import Optional, EmailStr, Field

class User(BaseModel):
    name: str = "Hariom"
    age: int
    email: Optional[EmailStr]
    cgpa: float = Field( ge=0.0, le=10.0, default=5, description="a decimal value representing CGPA of student")


new_user = {"name": "Alice"}

user = User(**new_user)

print(user.name) 