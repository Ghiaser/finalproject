import os
import sys
import time
import requests
import base64
import json
import random
import string
from pathlib import Path
import argparse

# Configuration
BASE_URL = "http://localhost:8000"
TEST_USERNAME = f"testuser_{random.randint(1000, 9999)}"
TEST_PASSWORD = "testpassword123"
TEST_INDEX = "test_index"

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def random_string(length=10):
    """Generate a random string"""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def print_status(message, status, details=None):
    """Print test status with color"""
    status_color = Colors.OKGREEN if status == "PASS" else Colors.FAIL
    print(f"{Colors.BOLD}{message}{Colors.ENDC} ... {status_color}{status}{Colors.ENDC}")
    if details and status == "FAIL":
        print(f"  {Colors.WARNING}Details: {details}{Colors.ENDC}")

def create_sample_files(data_dir):
    """Create sample text files for testing"""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a sample text file
    sample_text = f"""
    This is a sample text file for testing the API.
    It contains some random words for vector searching:
    mountain sky forest river ocean stars 
    technology computer science data model
    {random_string(20)}
    """
    
    with open(os.path.join(data_dir, "sample.txt"), "w") as f:
        f.write(sample_text)
    
    # Create another sample text file with different content
    sample_text2 = f"""
    Another sample file with different content.
    Space planets galaxy universe exploration
    Food recipe cooking delicious cuisine
    {random_string(20)}
    """
    
    with open(os.path.join(data_dir, "sample2.txt"), "w") as f:
        f.write(sample_text2)
        
    print(f"Created sample files in {data_dir}")
    return [
        os.path.join(data_dir, "sample.txt"),
        os.path.join(data_dir, "sample2.txt")
    ]

def test_create_user():
    """Test user creation endpoint"""
    url = f"{BASE_URL}/users"
    data = {
        "username": TEST_USERNAME,
        "password": TEST_PASSWORD
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 201:
            print_status("Create user", "PASS")
            return True
        else:
            print_status("Create user", "FAIL", f"Status code: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print_status("Create user", "FAIL", str(e))
        return False

def test_authenticate():
    """Test authentication with created user"""
    url = f"{BASE_URL}/users/me"
    
    try:
        response = requests.get(url, auth=(TEST_USERNAME, TEST_PASSWORD))
        if response.status_code == 200:
            print_status("Authenticate user", "PASS")
            return True
        else:
            print_status("Authenticate user", "FAIL", f"Status code: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print_status("Authenticate user", "FAIL", str(e))
        return False

def test_upload_files(file_paths):
    """Test file upload endpoint"""
    url = f"{BASE_URL}/upload"
    
    files = []
    for file_path in file_paths:
        files.append(("files", open(file_path, "rb")))
    
    try:
        response = requests.post(
            url, 
            files=files,
            auth=(TEST_USERNAME, TEST_PASSWORD)
        )
        
        for file in files:
            file[1].close()
            
        if response.status_code == 200:
            print_status("Upload files", "PASS")
            return True
        else:
            print_status("Upload files", "FAIL", f"Status code: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print_status("Upload files", "FAIL", str(e))
        return False

def test_list_files():
    """Test listing files endpoint"""
    url = f"{BASE_URL}/files"
    
    try:
        response = requests.get(url, auth=(TEST_USERNAME, TEST_PASSWORD))
        if response.status_code == 200:
            files = response.json().get("files", [])
            if len(files) >= 2:  # We expect at least our two sample files
                print_status("List files", "PASS")
                return True
            else:
                print_status("List files", "FAIL", f"Expected at least 2 files, got {len(files)}")
                return False
        else:
            print_status("List files", "FAIL", f"Status code: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print_status("List files", "FAIL", str(e))
        return False

def test_create_index():
    """Test creating an index"""
    url = f"{BASE_URL}/indexes"
    data = {
        "index_name": TEST_INDEX
    }
    
    try:
        response = requests.post(url, json=data, auth=(TEST_USERNAME, TEST_PASSWORD))
        if response.status_code == 201:
            print_status("Create index", "PASS")
            return True
        else:
            print_status("Create index", "FAIL", f"Status code: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print_status("Create index", "FAIL", str(e))
        return False

def test_list_indexes():
    """Test listing indexes endpoint"""
    url = f"{BASE_URL}/indexes"
    
    try:
        response = requests.get(url, auth=(TEST_USERNAME, TEST_PASSWORD))
        if response.status_code == 200:
            indexes = response.json().get("indexes", [])
            if TEST_INDEX in indexes:
                print_status("List indexes", "PASS")
                return True
            else:
                print_status("List indexes", "FAIL", f"Test index not found in {indexes}")
                return False
        else:
            print_status("List indexes", "FAIL", f"Status code: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print_status("List indexes", "FAIL", str(e))
        return False

def test_text_search():
    """Test text search endpoint"""
    url = f"{BASE_URL}/search/text"
    
    # Different query strings to test
    queries = [
        "mountain forest nature",
        "technology computer",
        "space planets",
        "food cooking"
    ]
    
    for query in queries:
        try:
            # The API expects a form field for index_name and JSON in the body for search_query
            search_query = {
                "query": query,
                "top_k": 2
            }
            
            response = requests.post(
                url,
                json=search_query,
                headers={"Content-Type": "application/json"},
                params={"index_name": TEST_INDEX},
                auth=(TEST_USERNAME, TEST_PASSWORD)
            )
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                if len(results) > 0:
                    print_status(f"Text search ({query})", "PASS")
                else:
                    print_status(f"Text search ({query})", "FAIL", "No results returned")
            else:
                print_status(f"Text search ({query})", "FAIL", f"Status code: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            print_status(f"Text search ({query})", "FAIL", str(e))
            return False
    
    return True

def test_image_upload_search():
    """Test image search with file upload endpoint"""
    # Check if 1.jpg exists in DATA directory
    image_path = os.path.join(os.getcwd(), "DATA", "1.jpg")
    if not os.path.exists(image_path):
        print_status("Image upload search", "SKIP", "1.jpg not found in DATA directory")
        return True
    
    url = f"{BASE_URL}/search/image"
    
    try:
        with open(image_path, 'rb') as img_file:
            files = {
                'image': ('1.jpg', img_file, 'image/jpeg')
            }
            
            data = {
                'index_name': TEST_INDEX,
                'top_k': 2
            }
            
            response = requests.post(
                url,
                files=files,
                data=data,
                auth=(TEST_USERNAME, TEST_PASSWORD)
            )
            
            if response.status_code == 200:
                print_status("Image upload search", "PASS")
                return True
            else:
                print_status("Image upload search", "FAIL", f"Status code: {response.status_code}, Response: {response.text}")
                return False
    except Exception as e:
        print_status("Image upload search", "FAIL", str(e))
        return False


def test_image_base64_search():
    """Test image search with base64 endpoint"""
    # Check if 2.jpg exists in DATA directory
    image_path = os.path.join(os.getcwd(), "DATA", "2.jpg")
    
    if not os.path.exists(image_path):
        # Fallback to simple test image if 2.jpg not available
        print_status("Image base64 search - using fallback pixel image", "INFO")
        pixel_data = base64.b64encode(bytes([0, 0, 255, 255])).decode('utf-8')  # Simple blue pixel
    else:
        # Use 2.jpg
        with open(image_path, 'rb') as img_file:
            pixel_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    url = f"{BASE_URL}/search/image/base64"
    
    try:
        # The API expects query parameters for this endpoint
        params = {
            "index_name": TEST_INDEX,
            "top_k": 2
        }
        
        data = {
            "image_base64": pixel_data
        }
        
        response = requests.post(
            url,
            json=data,
            params=params,
            auth=(TEST_USERNAME, TEST_PASSWORD)
        )
        
        if response.status_code == 200:
            print_status("Image base64 search", "PASS")
            return True
        else:
            print_status("Image base64 search", "FAIL", f"Status code: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print_status("Image base64 search", "FAIL", str(e))
        return False

def run_all_tests():
    """Run all API tests in sequence"""
    print(f"{Colors.HEADER}Starting API Tests{Colors.ENDC}")
    print("-" * 50)
    
    # Create sample files
    temp_dir = os.path.join(os.getcwd(), "temp_test_data")
    test_files = create_sample_files(temp_dir)
    
    # Test API endpoints
    if not test_create_user():
        print(f"{Colors.FAIL}User creation failed, aborting tests{Colors.ENDC}")
        return
    
    test_authenticate()
    test_upload_files(test_files)
    test_list_files()
    test_create_index()
    test_list_indexes()
    
    # Wait for a moment to ensure the index is fully created and ready
    print("Waiting for index to be ready...")
    time.sleep(2)
    
    test_text_search()
    test_image_upload_search()
    test_image_base64_search()
    
    print("-" * 50)
    print(f"{Colors.HEADER}Test Complete{Colors.ENDC}")

def check_api_available():
    """Check if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Secure CLIP API")
    parser.add_argument("--url", type=str, default="http://localhost:8000", 
                        help="Base URL for the API (default: http://localhost:8000)")
    args = parser.parse_args()
    
    BASE_URL = args.url
    
    if not check_api_available():
        print(f"{Colors.FAIL}API not available at {BASE_URL}. Make sure it's running.{Colors.ENDC}")
        print(f"Start the API with: python api.py")
        sys.exit(1)
    
    run_all_tests()