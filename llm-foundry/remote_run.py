#!/usr/bin/env python3
import subprocess
import json
import sys
import os

USER = ""
JOB = ""

def get_active_job():
    if JOB:
        return f"{JOB}-mpimaster-0.{JOB}"
    cmd = ["/workspace-SR008.nfs2/vbarinov/venvs/mls_venv/bin/mls", "job", "list"]
    try:
        output = subprocess.check_output(cmd, text=True)
        parsed_json = json.loads(output)
        
        job_list = []
        if isinstance(parsed_json, dict):
            for key, value in parsed_json.items():
                if isinstance(value, list):
                    job_list = value
                    break
            if not job_list:
                job_list = [parsed_json]
        elif isinstance(parsed_json, list):
            job_list = parsed_json
        else:
            print(f"Error: Unexpected JSON structure from MLS: {type(parsed_json)}")
            sys.exit(1)

        for j in job_list:
            if isinstance(j, dict):
                author_match = USER in j.get("job_desc", "") or USER in j.get("job_author", "")
                if author_match and j.get("status") == "Running":
                    return f"{j['job_name']}-mpimaster-0.{j['job_name']}"
                    
    except Exception as e:
        print(f"Failed to get job from MLS: {e}")
        sys.exit(1)
    
    print("Error: No running jobs found for user.")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python remote_run.py 'COMMAND'")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    user_command = " ".join(sys.argv[1:])
    # First go to directory with file
    command_to_run = f"cd '{script_dir}' && {user_command}"
    
    target_host = get_active_job()

    ssh_cmd = [
        "ssh", 
        "-i", "/home/jovyan/.ssh/ai0001071-01058", 
        "-p", "2222",
        "-o", "StrictHostKeyChecking=no", 
        "-q",
        f"user@{target_host}", 
        command_to_run
    ]

    try:
        subprocess.run(ssh_cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
