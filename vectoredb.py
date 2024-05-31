# vectoredb.py
import subprocess

def start_chromadb():
    # Assuming you have a script or command to start your ChromaDB server
    subprocess.run(["chromadb-server", "start"])

def stop_chromadb():
    # Assuming you have a script or command to stop your ChromaDB server
    subprocess.run(["chromadb-server", "stop"])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Control ChromaDB instance.")
    parser.add_argument('action', choices=['start', 'stop'], help="Action to perform: start or stop the ChromaDB instance.")
    
    args = parser.parse_args()

    if args.action == 'start':
        start_chromadb()
    elif args.action == 'stop':
        stop_chromadb()
