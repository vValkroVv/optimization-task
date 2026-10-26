#!/usr/bin/env python3

"""
Скрипт для запуска релиза через Pipeline Trigger Token
Usage: python trigger_release.py [--branch BRANCH]
Example: python trigger_release.py --branch main
"""
import os
import sys
import argparse
import requests
from typing import Optional


class ReleaseTrigger:
    def __init__(self):
        self.gitlab_url = "https://gitlab.ai.cloud.ru"
        self.project_id = "442"
        self.trigger_token = os.getenv("RELEASE_TRIGGER_TOKEN")

    def validate_token(self) -> bool:
        if not self.trigger_token:
            print(f"ERROR: TRIGGER_TOKEN environment variable is not set")
            print(f"Please set it first")
            print(f"export TRIGGER_TOKEN=your_trigger_token_here")
            return False
        return True

    def trigger_pipeline(self, branch: str = "main") -> bool:
        api_url = f"{self.gitlab_url}/api/v4/projects/{self.project_id}/trigger/pipeline"

        print(f"Starting release pipeline...")
        print(f"   Project: gigallms/llm-foundry (ID: {self.project_id})")
        print(f"   Branch: {branch}")
        print(f"   GitLab: {self.gitlab_url}")
        print()

        try:
            response = requests.post(
                api_url,
                data={
                    "token": self.trigger_token,
                    "ref": branch
                },
                timeout=30
            )

            if response.status_code == 201:
                data = response.json()
                pipeline_id = data.get("id")
                pipeline_url = data.get("web_url")

                print(f"Pipeline triggered successfully!")
                print(f"   Pipeline ID: {pipeline_id}")
                print(f"   URL: {pipeline_url}")
                print()
                print(f"You can monitor the pipeline at:")
                print(f"   {pipeline_url}")

                return True
            else:
                print(f"Failed to trigger pipeline")
                print(f"HTTP Status: {response.status_code}")

                try:
                    error_data = response.json()
                    message = error_data.get("message", "Unknown error")
                    print(f"Response:")
                    print(f"   {message}")
                except:
                    print(f"Response:")
                    print(f"   {response.text}")

                return False

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return False

    def run(self, branch: str = "main") -> int:
        if not self.validate_token():
            return 1

        if self.trigger_pipeline(branch):
            return 0
        else:
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="Trigger release pipeline via Pipeline Trigger Token"
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to trigger pipeline on (default: main)"
    )

    args = parser.parse_args()

    trigger = ReleaseTrigger()
    sys.exit(trigger.run(args.branch))


if __name__ == "__main__":
    main()