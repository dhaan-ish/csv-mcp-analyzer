#!/usr/bin/env python3
"""Wrapper script to run the complete MCP system with token check."""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path
from scripts.run_client import run as run_client

def check_token_exists():
    """Check if token.json exists in the project root or mcp_csv_analyzer directory."""
    token_paths = [
        Path("token.json"),
        Path("mcp_csv_analyzer/token.json")
    ]
    return any(path.exists() for path in token_paths)

def run_generate_token():
    """Run the token generation process."""
    print("=" * 60)
    print("Token not found. Running OAuth token generator...")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, "-m", "generate_token.generate_oauth_token"], 
                              check=True)
        if result.returncode == 0:
            print("\nToken generated successfully!")
            return True
    except subprocess.CalledProcessError as e:
        print(f"\nError generating token: {e}")
        return False
    except KeyboardInterrupt:
        print("\nToken generation cancelled.")
        return False
    
    return False

def run_mcp_system():
    """Run the MCP server and client."""
    server_process = None
    
    try:
        # Start the MCP server in the background
        print("\n" + "=" * 60)
        print("Starting MCP Server...")
        print("=" * 60)
        
        server_process = subprocess.Popen(
            [sys.executable, "-m", "mcp_csv_analyzer.server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give the server time to start
        print("Waiting for server to initialize...")
        time.sleep(3)
        
        # Check if server is still running
        if server_process.poll() is not None:
            print("Error: Server failed to start.")
            stdout, stderr = server_process.communicate()
            if stdout:
                print("Server output:", stdout)
            if stderr:
                print("Server error:", stderr)
            return
        
        print("Server started successfully!")
        
        # Run the client in the foreground
        print("\n" + "=" * 60)
        print("Starting MCP Client...")
        print("=" * 60)
        print("Note: Make sure you have configured your Azure OpenAI credentials in .env")
        print("=" * 60 + "\n")
        
        # Run the client directly using the imported function
        run_client()
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Clean up: terminate the server
        if server_process and server_process.poll() is None:
            print("\nStopping MCP Server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()
            print("Server stopped.")

def main():
    """Main entry point for the MCP runner."""
    print("MCP CSV Analyzer - Complete System Runner")
    print("=" * 60)
    
    # Check for token.json
    if not check_token_exists():
        print("Gmail OAuth token not found.")
        print("The token is required for email functionality.")
        
        response = input("\nDo you want to generate the token now? (y/n): ").strip().lower()
        
        if response == 'y':
            if not run_generate_token():
                print("\nFailed to generate token. Exiting.")
                return
        else:
            print("\nContinuing without email functionality...")
            print("You can generate the token later using: uv run generate_token")
    else:
        print("OAuth token found ✓")
    
    # Run the MCP system
    run_mcp_system()

if __name__ == '__main__':
    main() 