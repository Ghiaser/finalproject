import os
import json
import hashlib
import uuid
import secrets
import base64
from typing import Dict, List, Optional, Tuple

class UserManager:
    def __init__(self, users_file="users.json"):
        self.users_file = users_file
        self.users = self._load_users()

    def _load_users(self) -> Dict:
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        else:
            return {}

    def _save_users(self):
        # שומר לקובץ זמני ואז מחליף – הגנה מפני שיבוש
        temp_path = self.users_file + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(self.users, f, indent=2)
        os.replace(temp_path, self.users_file)

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, username: str, password: str) -> Tuple[bool, str]:
        if username in self.users:
            return False, "Username already exists"

        user_id = str(uuid.uuid4())
        user_salt = secrets.token_bytes(16).hex()

        os.makedirs(f"user_data/{username}", exist_ok=True)
        os.makedirs(f"user_data/{username}/indexes", exist_ok=True)
        os.makedirs(f"user_data/{username}/data", exist_ok=True)

        self.users[username] = {
            "id": user_id,
            "password_hash": self._hash_password(password),
            "salt": user_salt,
            "indexes": []
        }
        self._save_users()
        return True, "User created successfully"

    def authenticate(self, username: str, password: str) -> Tuple[bool, str]:
        if username not in self.users:
            return False, "User not found"

        if self.users[username]["password_hash"] != self._hash_password(password):
            return False, "Invalid password"

        return True, "Authentication successful"

    def get_user_folder(self, username: str) -> str:
        return f"user_data/{username}"

    def get_user_indexes(self, username: str) -> List[str]:
        return self.users[username].get("indexes", [])

    def add_user_index(self, username: str, index_name: str) -> bool:
        if username not in self.users:
            return False

        if "indexes" not in self.users[username]:
            self.users[username]["indexes"] = []

        if index_name not in self.users[username]["indexes"]:
            self.users[username]["indexes"].append(index_name)
            self._save_users()

        return True

    def remove_user_index(self, username: str, index_name: str) -> bool:
        if username not in self.users:
            return False

        if "indexes" in self.users[username] and index_name in self.users[username]["indexes"]:
            self.users[username]["indexes"].remove(index_name)
            self._save_users()

        return True

    def get_user_salt(self, username: str) -> Optional[bytes]:
        if username not in self.users:
            return None
        salt_hex = self.users[username].get("salt")
        if not salt_hex:
            return None
        return bytes.fromhex(salt_hex)

    def get_user_fernet_key(self, username: str, password: str) -> bytes:
        salt = self.get_user_salt(username)
        if salt is None:
            raise ValueError("Salt not found for user")
        derived = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100_000)
        return base64.urlsafe_b64encode(derived)
