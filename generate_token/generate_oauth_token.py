#!/usr/bin/env python3
"""
OAuth Token Generation Script for Gmail API

This script helps generate and store OAuth tokens for Gmail API access.
It reads client credentials and performs the OAuth flow to obtain access tokens.
"""

import os
import json
import sys
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv

load_dotenv()

SCOPES = [
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.compose',
    'https://www.googleapis.com/auth/gmail.modify'
]

def load_credentials_from_file(credentials_file):
    """Load client credentials from a JSON file."""
    try:
        with open(credentials_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Credentials file '{credentials_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{credentials_file}'.")
        sys.exit(1)

def get_credentials_from_input():
    """Get credentials from user input."""
    print("Enter your OAuth2 credentials:")
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Error: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in environment variables.")
        sys.exit(1)
    
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "project_id": "mcp-server-463309",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "redirect_uris": ["http://localhost"]
        }
    }

def generate_token(credentials_data, token_file='token.json'):
    """Generate and save OAuth token."""
    creds = None
    
    if os.path.exists(token_file):
        print(f"Token file '{token_file}' already exists.")
        response = input("Do you want to regenerate the token? (y/n): ").strip().lower()
        if response != 'y':
            print("Using existing token.")
            return
        else:
            try:
                creds = Credentials.from_authorized_user_file(token_file, SCOPES)
            except:
                pass
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired token...")
            creds.refresh(Request())
        else:
            temp_creds_file = 'temp_credentials.json'
            with open(temp_creds_file, 'w') as f:
                json.dump(credentials_data, f)
            
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    temp_creds_file, SCOPES)
                
                print("\nStarting OAuth2 flow...")
                print("A browser window will open for authentication.")
                print("Please log in and grant the required permissions.")
                
                creds = flow.run_local_server(port=0)
                
                os.remove(temp_creds_file)
                
            except Exception as e:
                if os.path.exists(temp_creds_file):
                    os.remove(temp_creds_file)
                print(f"Error during OAuth flow: {e}")
                sys.exit(1)
    
    with open(token_file, 'w') as token:
        token.write(creds.to_json())
    
    print(f"\nToken successfully saved to '{token_file}'")
    print("You can now use this token for Gmail API access.")
    
    token_info = json.loads(creds.to_json())
    print("\nToken information:")
    print(f"- Token type: {token_info.get('token', 'N/A')[:20]}...")
    print(f"- Refresh token: {'Present' if token_info.get('refresh_token') else 'Not present'}")
    print(f"- Expiry: {token_info.get('expiry', 'N/A')}")
    print(f"- Scopes: {', '.join(token_info.get('scopes', []))}")

def main():
    """Main function to handle the token generation process."""
    print("Gmail OAuth Token Generator")
    print("==========================\n")
    credentials_data = get_credentials_from_input()
    
    token_file = 'token.json'
    
    generate_token(credentials_data, token_file)

if __name__ == "__main__":
    main() 