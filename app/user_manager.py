# file: app/user_manager.py

import os
import json
import hashlib
import uuid
import secrets
import base64
from typing import Dict, List, Optional, Tuple

# ====================================================================================
# If you want to allow monkeypatching for test fixes, store BASE_USER_DATA here.
# You can set the environment variable BASE_USER_DATA; otherwise, the default is "app/user_data".
# ====================================================================================
BASE_USER_DATA = os.getenv("BASE_USER_DATA", "app/user_data")


class UserManager:
    def __init__(self, users_file: Optional[str] = None):
        """
        Class for user management:
         - Create a new user (including creating data and indexes directories)
         - Authenticate password
         - Produce a Fernet key for file encryption/decryption
         - Manage the list of index names
        """
        # Default: save users.json under BASE_USER_DATA
        if users_file:
            self.users_file = users_file
        else:
            # Ensure BASE_USER_DATA directory exists
            os.makedirs(BASE_USER_DATA, exist_ok=True)
            self.users_file = os.path.join(BASE_USER_DATA, "users.json")

        # Load users from file (if exists)
        self.users = self._load_users()

    def _load_users(self) -> Dict:
        """
        Load the user mapping from users.json.
        If the file does not exist or is corrupted, return an empty dict.
        """
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        else:
            return {}

    def _save_users(self):
        """
        Save the user data to users.json.
        For reliability, write to a temporary file first, then replace.
        """
        temp_path = self.users_file + ".tmp"
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        with open(temp_path, 'w') as f:
            json.dump(self.users, f, indent=2)
        os.replace(temp_path, self.users_file)

    def _hash_password(self, password: str) -> str:
        """
        Create a SHA256 hash of the password.
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Create a new user:
         - Check if the username already exists; if so, return False.
         - Generate a random user_id and salt.
         - Create data and indexes directories under BASE_USER_DATA/<username>/...
         - Add the user to self.users and save to users.json.
        """
        if username in self.users:
            return False, "Username already exists"

        # Unique user identifier
        user_id = str(uuid.uuid4())
        # Unique salt for file encryption
        user_salt = secrets.token_bytes(16).hex()

        # Ensure BASE_USER_DATA directories exist
        user_root = os.path.join(BASE_USER_DATA, username)
        data_dir = os.path.join(user_root, "data")
        indexes_dir = os.path.join(user_root, "indexes")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(indexes_dir, exist_ok=True)

        # Structure that will be saved in users.json
        self.users[username] = {
            "id": user_id,
            "password_hash": self._hash_password(password),
            "salt": user_salt,
            "indexes": []
        }
        self._save_users()
        return True, "User created successfully"

    def authenticate(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Authenticate a user by checking the username and password hash.
        Returns (True, message) if password is correct; otherwise (False, error).
        """
        if username not in self.users:
            return False, "User not found"

        if self.users[username]["password_hash"] != self._hash_password(password):
            return False, "Invalid password"

        return True, "Authentication successful"

    def get_user_folder(self, username: str) -> str:
        """
        Return the path to the user's root folder (BASE_USER_DATA/<username>).
        """
        return os.path.join(BASE_USER_DATA, username)

    def get_user_data_dir(self, username: str) -> str:
        """
        Return the path to the user's data directory (BASE_USER_DATA/<username>/data).
        """
        return os.path.join(BASE_USER_DATA, username, "data")

    def get_user_indexes_dir(self, username: str) -> str:
        """
        Return the path to the folder where all the user's index files are stored
        (BASE_USER_DATA/<username>/indexes).
        """
        return os.path.join(BASE_USER_DATA, username, "indexes")

    def get_user_indexes(self, username: str) -> List[str]:
        """
        Return the list of index names that the user has created (or [] if none).
        """
        return self.users.get(username, {}).get("indexes", [])

    def add_user_index(self, username: str, index_name: str) -> bool:
        """
        Add index_name to the user's list of indexes and save to users.json.
        If the user does not exist → return False.
        """
        if username not in self.users:
            return False

        if "indexes" not in self.users[username]:
            self.users[username]["indexes"] = []

        if index_name not in self.users[username]["indexes"]:
            self.users[username]["indexes"].append(index_name)
            self._save_users()

        return True

    def remove_user_index(self, username: str, index_name: str) -> bool:
        """
        Remove index_name from the user's list of indexes and save to users.json.
        If the user does not exist → return False.
        """
        if username not in self.users:
            return False

        if "indexes" in self.users[username] and index_name in self.users[username]["indexes"]:
            self.users[username]["indexes"].remove(index_name)
            self._save_users()

        return True

    def get_user_salt(self, username: str) -> Optional[bytes]:
        """
        Return the user's salt (bytes). If the user does not exist or has no salt → return None.
        """
        if username not in self.users:
            return None
        salt_hex = self.users[username].get("salt")
        if not salt_hex:
            return None
        return bytes.fromhex(salt_hex)

    def get_user_fernet_key(self, username: str, password: str) -> bytes:
        """
        Derive a Fernet key based on PBKDF2-HMAC-SHA256.
        Assumes the salt is already stored on the user. If no salt → ValueError.
        Returns the key in base64-url-safe format.
        """
        salt = self.get_user_salt(username)
        if salt is None:
            raise ValueError("Salt not found for user")
        derived = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100_000)
        return base64.urlsafe_b64encode(derived)
