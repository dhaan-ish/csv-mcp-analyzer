#!/usr/bin/env python3
"""Wrapper script to run the MCP client."""

import asyncio
import sys
from client.app import main

def run():
    """Run the MCP client application."""
    asyncio.run(main())

if __name__ == '__main__':
    run() 