#!/usr/bin/env python3
"""
Network connection information display for PANSINAYAN.

PANSINAYAN - Filipino Sign Language Recognition System
"Where Every Sign Gets Attention"

This utility displays connection information for accessing the PANSINAYAN
Streamlit application from different devices on your network.
"""

import socket
import sys
import platform

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        # Connect to external DNS to determine local network IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return None

def print_network_info(port=8501):
    """Print connection information for the PANSINAYAN Streamlit app."""
    local_ip = get_local_ip()
    hostname = socket.gethostname()
    os_name = platform.system()
    
    print("\n" + "="*70)
    print("  🤟 PANSINAYAN - Filipino Sign Language Recognition")
    print("  " + "-"*66)
    print("  Where Every Sign Gets Attention")
    print("="*70)
    
    print("\n📡 CONNECTION INFORMATION:")
    print(f"   • Hostname: {hostname}")
    print(f"   • Operating System: {os_name}")
    if local_ip:
        print(f"   • Local IP: {local_ip}")
    
    print("\n🌐 ACCESS URLs:")
    print(f"   📍 LOCAL:   http://localhost:{port}")
    if local_ip:
        print(f"   📱 NETWORK: http://{local_ip}:{port}")
        print(f"      (Use this URL to access from other devices on your network)")
    else:
        print(f"   📱 NETWORK: Check Streamlit's 'Network URL' in the terminal output")
    
    print("\n🔧 ADVANCED OPTIONS:")
    print(f"   • Cloudflare Tunnel: cloudflared tunnel --url http://localhost:{port}")
    print(f"   • Custom Port: streamlit run run_app.py --server.port <PORT>")
    
    print("\n💡 USAGE:")
    print(f"   Start the app with: streamlit run run_app.py")
    print(f"   Then access using any of the URLs above")
    
    print("\n📚 MODELS:")
    print("   • SignTransformer (Multi-Head Attention)")
    print("   • InceptionV3+GRU (CNN-RNN Hybrid)")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    # Get port from command line argument if provided (default: 8501 - Streamlit default)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    print_network_info(port)
