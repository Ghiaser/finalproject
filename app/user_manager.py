import os
import json
import hashlib
import uuid
from typing import Dict, List, Optional, Tuple

# Simple file-based user management system
class UserManager:
    def __init__(self, users_file="users.json"):
        self.users_file = users_file
        self.users = self._load_users()
        
    def _load_users(self) -> Dict:
        """Load users from file or create empty user store"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        else:
            return {}
            
    def _save_users(self):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
            
    def _hash_password(self, password: str) -> str:
        """Create password hash"""
        return hashlib.sha256(password.encode()).hexdigest()
        
    def create_user(self, username: str, password: str) -> Tuple[bool, str]:
        """Create a new user"""
        if username in self.users:
            return False, "Username already exists"
            
        # Create user folders
        user_id = str(uuid.uuid4())
        os.makedirs(f"user_data/{username}", exist_ok=True)
        os.makedirs(f"user_data/{username}/indexes", exist_ok=True)
        os.makedirs(f"user_data/{username}/data", exist_ok=True)
        
        # Store user
        self.users[username] = {
            "id": user_id,
            "password_hash": self._hash_password(password),
            "indexes": []
        }
        self._save_users()
        
        return True, "User created successfully"
        
    def authenticate(self, username: str, password: str) -> Tuple[bool, str]:
        """Authenticate a user"""
        if username not in self.users:
            return False, "User not found"
            
        if self.users[username]["password_hash"] != self._hash_password(password):
            return False, "Invalid password"
            
        return True, "Authentication successful"
        
    def get_user_folder(self, username: str) -> str:
        """Get the user's data folder"""
        return f"user_data/{username}"
        
    def get_user_indexes(self, username: str) -> List[str]:
        """Get list of user's indexes"""
        return self.users[username].get("indexes", [])
        
    def add_user_index(self, username: str, index_name: str) -> bool:
        """Add index to user's list"""
        if username not in self.users:
            return False
            
        if "indexes" not in self.users[username]:
            self.users[username]["indexes"] = []
            
        if index_name not in self.users[username]["indexes"]:
            self.users[username]["indexes"].append(index_name)
            self._save_users()
            
        return True
        
    def remove_user_index(self, username: str, index_name: str) -> bool:
        """Remove index from user's list"""
        if username not in self.users:
            return False
            
        if "indexes" in self.users[username] and index_name in self.users[username]["indexes"]:
            self.users[username]["indexes"].remove(index_name)
            self._save_users()
            
        return True