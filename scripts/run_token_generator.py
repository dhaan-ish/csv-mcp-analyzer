#!/usr/bin/env python3
"""Wrapper script to run the OAuth token generator."""

import sys
import runpy

def main():
    """Run the OAuth token generator."""
    sys.argv[0] = 'generate_token.generate_oauth_token'
    runpy.run_module('generate_token.generate_oauth_token', run_name='__main__')

if __name__ == '__main__':
    main() 