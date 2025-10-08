#!/usr/bin/env python3
"""Script to display network connection information for the FSLR Streamlit app."""

import socket
import sys

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return None

def print_network_info(port=8081):
    """Print connection information for the Streamlit app."""
    local_ip = get_local_ip()
    
    print("\n" + "="*60)
    print("  FSLR STREAMLIT APP - CONNECTION GUIDE")
    print("="*60)
    print(f"\n📍 LOCAL ACCESS: http://localhost:{port}")
    if local_ip:
        print(f"📱 NETWORK ACCESS: http://{local_ip}:{port}")
    else:
        print(f"📱 NETWORK ACCESS: Check Streamlit's 'Network URL' in terminal")
    print(f"🌐 CLOUDFLARE TUNNEL: cloudflared tunnel --url http://localhost:{port}")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Get port from command line argument if provided
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    print_network_info(port)
