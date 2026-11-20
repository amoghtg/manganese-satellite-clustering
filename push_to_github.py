#!/usr/bin/env python3
"""
Automated GitHub Repository Creator and Code Pusher for SIH26009.
Creates a remote GitHub repository via GitHub REST API and pushes all committed project files.
"""

import os
import sys
import argparse
import subprocess
import requests

def create_and_push(username: str, token: str, repo_name: str = "manganese-satellite-clustering", private: bool = False):
    print(f"\n1. Connecting to GitHub API for user '{username}'...")
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "name": repo_name,
        "description": "SIH26009: K-Means Clustering of Satellite Mineral-Signature Data to Identify Manganese-Rich Zones",
        "private": private,
        "auto_init": False
    }

    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 201:
        repo_data = response.json()
        clone_url = repo_data["clone_url"]
        html_url = repo_data["html_url"]
        print(f" Repository created successfully on GitHub: {html_url}")
    elif response.status_code == 422:
        # Repository already exists
        print(f"ℹ️ Repository '{repo_name}' already exists under user '{username}'. Using existing repository.")
        html_url = f"https://github.com/{username}/{repo_name}"
        clone_url = f"https://github.com/{username}/{repo_name}.git"
    else:
        print(f" Failed to create repository (Status {response.status_code}): {response.text}")
        sys.exit(1)

    print("\n2. Configuring local Git remote...")
    # Remove existing origin if any
    subprocess.run(["git", "remote", "remove", "origin"], stderr=subprocess.DEVNULL)

    # Use authenticated URL for push
    auth_url = clone_url.replace("https://", f"https://{username}:{token}@")
    subprocess.run(["git", "remote", "add", "origin", auth_url], check=True)

    print("\n3. Pushing branch 'main' to GitHub...")
    push_res = subprocess.run(["git", "push", "-u", "origin", "main"], capture_output=True, text=True)

    if push_res.returncode == 0:
        # Clean remote URL to not store token in git config
        subprocess.run(["git", "remote", "set-url", "origin", clone_url], check=True)
        print("\n" + "=" * 60)
        print(f" SUCCESS! All files and documents pushed to GitHub:")
        print(f" {html_url}")
        print("=" * 60 + "\n")
    else:
        print(f" Git push failed:\n{push_res.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GitHub repo and push project.")
    parser.add_argument("--username", "-u", type=str, help="GitHub username")
    parser.add_argument("--token", "-t", type=str, help="GitHub Personal Access Token (PAT)")
    parser.add_argument("--repo", "-r", type=str, default="manganese-satellite-clustering", help="Repository name")
    parser.add_argument("--private", action="store_true", help="Make repository private")

    args = parser.parse_args()

    username = args.username or os.getenv("GITHUB_USERNAME")
    token = args.token or os.getenv("GITHUB_TOKEN")

    if not username:
        username = input("Enter your GitHub username: ").strip()
    if not token:
        import getpass
        token = getpass.getpass("Enter your GitHub Personal Access Token (classic with repo scope or fine-grained): ").strip()

    create_and_push(username, token, args.repo, args.private)
