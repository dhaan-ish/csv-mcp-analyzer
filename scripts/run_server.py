#!/usr/bin/env python3
"""Wrapper script to run the MCP server."""

import sys
import runpy

def main():
    """Run the MCP CSV Analyzer server."""
    sys.argv[0] = 'mcp_csv_analyzer.server'
    runpy.run_module('mcp_csv_analyzer.server', run_name='__main__')

if __name__ == '__main__':
    main() 