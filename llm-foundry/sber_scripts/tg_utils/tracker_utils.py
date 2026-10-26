import json
import sys
from pathlib import Path
from typing import List, Union, Any
from datetime import datetime

DAEMON_STATE_PATHS = {
    "pretrain:sr004": "/home/jovyan/daemon_state",
    "pretrain:sr008": "/home/jovyan/daemon_state",
    "vision:sr008": "/home/jovyan/daemon_state",
    "papaya:o1": "/mnt/nfs/01-home/infidelis/transfer2cloud/o1/daemon_state",
    "papaya:o4": "/mnt/nfs/01-home/loggerhead/transfer2cloud/o4/nfsbot",
    "papaya:o5": "/mnt/nfs/01-home/infidelis/transfer2cloud/o5/nfsbot",
}

REGION_ALIASES = {
    "SR004": "pretrain:sr004",
    "SR008": "pretrain:sr008",
    "o1": "papaya:o1",
    "o4": "papaya:o4",
    "o5": "papaya:o5",
}


def parse_region(region: str) -> tuple:
    if ':' in region:
        parts = region.split(':', 1)
        return parts[0], parts[1]

    if region in REGION_ALIASES:
        full_name = REGION_ALIASES[region]
        return full_name.split(':', 1)

    raise ValueError(f"Unknown region: {region}")


def get_daemon_state_path(workspace: str, cluster: str) -> Path:
    full_name = f"{workspace}:{cluster}"

    if full_name not in DAEMON_STATE_PATHS:
        raise ValueError(f"Unknown workspace:cluster: {full_name}")

    return Path(DAEMON_STATE_PATHS[full_name])


def normalize_logs_path(logs_path: str, workspace: str, cluster: str) -> str:
    full_prefix = f"{workspace}:{cluster}:"

    while logs_path.startswith(full_prefix):
        logs_path = logs_path[len(full_prefix):]

    if ':' in logs_path:
        parts = logs_path.split(':', 2)
        if len(parts) == 3 and parts[0] and parts[1]:
            print(f"Warning: logs_path contains cluster prefix from different cluster: {logs_path}")
            print(f"         Removing prefix '{parts[0]}:{parts[1]}:'")
            return parts[2]

    return logs_path


def get_nested_value(data: Any, path: str, default=None):
    keys = path.split('.')
    value = data

    for key in keys:
        if value is None:
            return default

        if isinstance(value, dict):
            value = value.get(key)
        elif hasattr(value, key):
            value = getattr(value, key)
        else:
            return default

    return value if value is not None else default


def resolve_variables(config: dict) -> dict:
    import re

    def resolve_value(value, context):
        if not isinstance(value, str):
            return value

        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)

        if not matches:
            return value

        result = value
        for var_name in matches:
            if var_name in context:
                var_value = context[var_name]
                if isinstance(var_value, str) and '${' in var_value:
                    var_value = resolve_value(var_value, context)
                result = result.replace(f'${{{var_name}}}', str(var_value))

        return result

    def resolve_dict(d, context):
        resolved = {}
        for key, value in d.items():
            if isinstance(value, dict):
                resolved[key] = resolve_dict(value, context)
            elif isinstance(value, list):
                resolved[key] = [
                    resolve_dict(item, context) if isinstance(item, dict)
                    else resolve_value(item, context)
                    for item in value
                ]
            else:
                resolved[key] = resolve_value(value, context)
        return resolved

    context = {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool))}

    resolved_config = resolve_dict(config, context)

    context.update({k: v for k, v in resolved_config.items() if isinstance(v, (str, int, float, bool))})
    resolved_config = resolve_dict(resolved_config, context)

    return resolved_config


def extract_logs_path_from_config(config_path: str, run_name: str = None) -> str:
    try:
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load config: {e}")

    resolved_config = resolve_variables(config)

    log_dir = get_nested_value(resolved_config, 'loggers.tensorboard.log_dir')

    if log_dir:
        if not run_name:
            run_name = get_nested_value(resolved_config, 'run_name', 'llm')

        logs_path = f"{log_dir}/{run_name}"
        logs_path = logs_path.replace('\\', '/')
        if len(logs_path) > 2 and logs_path[1] == ':':
            logs_path = logs_path[2:]
        return logs_path

    raise ValueError(
        "Could not find logs path in config. "
        "Expected 'loggers.tensorboard.log_dir'"
    )


def extract_logs_path_from_merged_config(merged_config: dict, run_name: str = None) -> str:
    resolved_config = resolve_variables(merged_config)

    log_dir = get_nested_value(resolved_config, 'loggers.tensorboard.log_dir')

    if log_dir:
        if not run_name:
            run_name = get_nested_value(resolved_config, 'run_name', 'llm')

        logs_path = f"{log_dir}/{run_name}"
        logs_path = logs_path.replace('\\', '/')
        if len(logs_path) > 2 and logs_path[1] == ':':
            logs_path = logs_path[2:]
        return logs_path

    raise ValueError(
        "Could not find logs path in config. "
        "Expected 'loggers.tensorboard.log_dir'"
    )


def write_auto_track_request(
        workspace: str,
        cluster: str,
        users: Union[str, List[str]],
        logs_path: str,
        yaml_path: str = None,
        script_path: str = None
) -> bool:
    print(f"\n{'='*70}")
    print(f"📝 write_auto_track_request() called")
    print(f"{'='*70}")

    try:
        print(f"\n1️⃣  Input parameters:")
        print(f"   workspace = {repr(workspace)}")
        print(f"   cluster = {repr(cluster)}")
        print(f"   users = {users}")
        print(f"   logs_path = {repr(logs_path)}")
        print(f"   yaml_path = {repr(yaml_path)}")
        print(f"   script_path = {repr(script_path)}")

        # Get daemon state directory
        print(f"\n2️⃣  Resolving daemon state directory...")
        daemon_state_dir = get_daemon_state_path(workspace, cluster)
        print(f"   Daemon state dir: {daemon_state_dir}")

        # Normalize logs_path
        print(f"\n3️⃣  Normalizing logs_path...")
        print(f"   Original: {repr(logs_path)}")
        logs_path = normalize_logs_path(logs_path, workspace, cluster)
        print(f"   Normalized: {repr(logs_path)}")

        if logs_path.startswith(f"{workspace}:{cluster}:"):
            print(f"   ⚠️  WARNING: logs_path still contains cluster prefix!")
            print(f"   This may indicate a problem with normalize_logs_path()")
        else:
            print(f"   ✅ logs_path is clean (no cluster prefix)")

        # Prepare users list
        print(f"\n4️⃣  Preparing users list...")
        if isinstance(users, str):
            users = [users]
        print(f"   Users: {users} (type: {type(users).__name__})")

        # Create request data
        print(f"\n5️⃣  Creating auto-track request data...")
        auto_track_data = {
            "users": users,
            "logs_path": logs_path,
            "timestamp": datetime.now().isoformat()
        }

        if yaml_path:
            auto_track_data["yaml_path"] = yaml_path
            print(f"   ✅ yaml_path added")
        if script_path:
            auto_track_data["script_path"] = script_path
            print(f"   ✅ script_path added")

        print(f"   Request data structure:")
        for key, value in auto_track_data.items():
            if key == 'timestamp':
                print(f"      {key}: {value}")
            else:
                print(f"      {key}: {repr(value)}")

        # Ensure directory exists
        auto_track_file = daemon_state_dir / "auto_track_requests.txt"

        print(f"\n6️⃣  Checking daemon state directory...")
        print(f"   Target file: {auto_track_file}")

        if not daemon_state_dir.exists():
            print(f"   ⚠️  Directory does not exist: {daemon_state_dir}")
            print(f"   Attempting to create it...")
            try:
                daemon_state_dir.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Directory created successfully")
            except Exception as mkdir_err:
                print(f"   ❌ Failed to create directory: {mkdir_err}")
                return False
        else:
            print(f"   ✅ Directory exists")

        # Check write permissions
        if not daemon_state_dir.is_dir():
            print(f"   ❌ Path is not a directory: {daemon_state_dir}")
            return False

        # Write to file
        print(f"\n7️⃣  Writing to file...")

        # Check if file exists and show its current state
        if auto_track_file.exists():
            file_size = auto_track_file.stat().st_size
            print(f"   File exists, current size: {file_size} bytes")

            # Count existing lines
            with open(auto_track_file, 'r', encoding='utf-8') as f:
                existing_lines = len(f.readlines())
            print(f"   Existing requests in file: {existing_lines}")
        else:
            print(f"   File does not exist, will be created")

        # Append new request
        with open(auto_track_file, 'a', encoding='utf-8') as f:
            json_line = json.dumps(auto_track_data)
            f.write(json_line + '\n')
            f.flush()  # Ensure it's written immediately

        print(f"   ✅ Request written to file")

        # Verify write
        print(f"\n8️⃣  Verifying write operation...")
        if auto_track_file.exists():
            new_size = auto_track_file.stat().st_size
            print(f"   New file size: {new_size} bytes")

            # Read back the last line
            with open(auto_track_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                last_line = lines[-1].strip() if lines else None

            if last_line:
                try:
                    verified_data = json.loads(last_line)
                    print(f"   ✅ Last line is valid JSON")
                    print(f"   Verified logs_path: {verified_data.get('logs_path')}")

                    if verified_data.get('logs_path') == logs_path:
                        print(f"   ✅ logs_path matches what we wrote")
                    else:
                        print(f"   ⚠️  logs_path mismatch!")
                except json.JSONDecodeError as e:
                    print(f"   ⚠️  Last line is not valid JSON: {e}")
        else:
            print(f"   ⚠️  File does not exist after write!")

        # Success summary
        print(f"\n{'='*70}")
        print(f"✅ AUTO-TRACK REQUEST CREATED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"   Cluster: {workspace}:{cluster}")
        print(f"   Users: {', '.join(users)}")
        print(f"   Logs path: {logs_path}")
        print(f"   File: {auto_track_file}")
        print(f"{'='*70}")

        return True

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ FAILED TO CREATE AUTO-TRACK REQUEST")
        print(f"{'='*70}")
        print(f"   Exception type: {type(e).__name__}")
        print(f"   Exception message: {e}")
        print(f"\n   Full traceback:")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}")
        return False


def auto_track_from_region(
        region: str,
        users: Union[str, List[str]],
        logs_path: str,
        yaml_path: str = None,
        script_path: str = None
) -> bool:
    """Create auto-track request using region string."""
    try:
        workspace, cluster = parse_region(region)

        return write_auto_track_request(
            workspace=workspace,
            cluster=cluster,
            users=users,
            logs_path=logs_path,
            yaml_path=yaml_path,
            script_path=script_path
        )
    except Exception as e:
        print(f"Failed to create auto-track request: {e}")
        import traceback
        traceback.print_exc()
        return False


def auto_track_from_config(
        region: str,
        users: Union[str, List[str]],
        config_path: str,
        run_name: str = None
) -> bool:
    """
    Create auto-track request by extracting logs path from config.
    Works with both cloud configs (loggers.tensorboard.log_dir)
    and papaya configs (save_folder).
    """
    try:
        logs_path = extract_logs_path_from_config(config_path, run_name)

        import os
        script_path = os.path.abspath(sys.argv[0]) if sys.argv else None

        return auto_track_from_region(
            region=region,
            users=users,
            logs_path=logs_path,
            yaml_path=os.path.abspath(config_path),
            script_path=script_path
        )

    except Exception as e:
        print(f"Failed to auto-track from config: {e}")
        import traceback
        traceback.print_exc()
        return False


def auto_track_from_merged_config(
        region: str,
        users: Union[str, List[str]],
        merged_config: dict,
        config_path: str
) -> bool:
    """
    Create auto-track request from merged config dict.
    Resolves ${variable} references automatically.
    """
    try:
        logs_path = extract_logs_path_from_merged_config(merged_config)

        import os
        script_path = os.path.abspath(sys.argv[0]) if sys.argv else None

        return auto_track_from_region(
            region=region,
            users=users,
            logs_path=logs_path,
            yaml_path=os.path.abspath(config_path),
            script_path=script_path
        )

    except Exception as e:
        print(f"Failed to auto-track from merged config: {e}")
        import traceback
        traceback.print_exc()
        return False


def auto_track(
        users: Union[str, List[str]],
        yaml_path: Union[Path, str],
        resume_timestamp: int = None,
        auto_track_path: Path = None,
        run_name: str = None
) -> None:
    try:
        import os
        region = os.environ.get('REGION', 'SR004')

        print(f"🔍 Auto-tracking with region: {region}")

        auto_track_from_config(
            region=region,
            users=users,
            config_path=str(yaml_path),
            run_name=run_name
        )

    except Exception as e:
        print(f"Auto-track failed: {e}")
        import traceback
        traceback.print_exc()


def run_auto_track(
        region: str,
        submission_type: str,
        code_name=None,
        user_login=None,
        config_path: str = None,
        config_name: str = None,
        default_config_name: str = None,
        merged_config_dict: dict = None
) -> bool:
    """
    Run automatic experiment tracking.

    Args:
        region: Region identifier (e.g., 'SR004', 'pretrain:sr004')
        submission_type: Either 'slurm' or 'cloud'
        code_name: Username(s) for SLURM submissions
        user_login: Username(s) for cloud submissions
        config_path: Path to config file
        config_name: Name of the config being used
        default_config_name: Optional default config name
        merged_config_dict: Optional merged config dictionary

    Returns:
        bool: True if auto-tracking succeeded, False otherwise
    """
    print("\n" + "=" * 80)
    print("AUTO-TRACK: Starting automatic experiment tracking")
    print("=" * 80)

    try:
        # Determine users based on submission type
        if submission_type == "slurm":
            users = [code_name] if isinstance(code_name, str) else code_name
            print(f"Submission type: SLURM")
            print(f"Users (CODE_NAME): {users}")
        else:
            users = [user_login] if isinstance(user_login, str) else user_login
            print(f"Submission type: CLOUD")
            print(f"Users (USER_LOGIN): {users}")

        # Display region information
        _print_region_info(region)

        # Display config information
        _print_config_info(config_path, config_name, default_config_name)

        # Extract and display logs path
        extracted_logs_path = _extract_and_display_logs_path(
            config_path, default_config_name, merged_config_dict
        )

        # Execute auto-tracking
        success = _execute_auto_track(
            region=region,
            users=users,
            config_path=config_path,
            default_config_name=default_config_name,
            merged_config_dict=merged_config_dict
        )

        # Display results
        _print_results(success, region)

        return success

    except Exception as e:
        _print_error(e)
        return False
    finally:
        print("=" * 80)
        print("🔍 AUTO-TRACK: Section completed")
        print("=" * 80 + "\n")


def _print_region_info(region: str) -> None:
    """Print region configuration details."""
    print(f"\nRegion configuration:")
    print(f"   REGION: {region}")

    try:
        from tracker_utils import parse_region, get_daemon_state_path

        workspace, cluster = parse_region(region)
        daemon_path = get_daemon_state_path(workspace, cluster)
        print(f"   Workspace: {workspace}")
        print(f"   Cluster: {cluster}")
        print(f"   Full name: {workspace}:{cluster}")
        print(f"   Daemon state directory: {daemon_path}")
    except Exception as region_err:
        print(f"   Could not parse region details: {region_err}")


def _print_config_info(
        config_path: str,
        config_name: str = None,
        default_config_name: str = None
) -> None:
    """Print configuration file information."""
    import os

    print(f"\nConfiguration files:")
    print(f"   Config path: {config_path}")

    if default_config_name:
        print(f"   Default config: {default_config_name}")
        print(f"   Editable config: {config_name}")
        print(f"   Merged config: {config_path}")
    else:
        print(f"   Direct config: {config_name}")

    if os.path.exists(config_path):
        print(f"   Config file exists")
    else:
        print(f"   Config file NOT FOUND!")


def _extract_and_display_logs_path(
        config_path: str,
        default_config_name: str = None,
        merged_config_dict: dict = None
) -> str:
    """Extract and display logs path from config."""
    print(f"\nExtracting logs path from config...")

    try:
        if default_config_name and merged_config_dict:
            from tracker_utils import extract_logs_path_from_merged_config
            extracted_logs_path = extract_logs_path_from_merged_config(merged_config_dict)
        else:
            from tracker_utils import extract_logs_path_from_config
            extracted_logs_path = extract_logs_path_from_config(config_path)

        print(f"   Extracted logs_path: {extracted_logs_path}")
        return extracted_logs_path
    except Exception as extract_err:
        print(f"   Could not extract logs_path for preview: {extract_err}")
        return None


def _execute_auto_track(
        region: str,
        users: list,
        config_path: str,
        default_config_name: str = None,
        merged_config_dict: dict = None
) -> bool:
    """Execute the actual auto-tracking operation."""
    if default_config_name and merged_config_dict:
        from tracker_utils import auto_track_from_merged_config
        return auto_track_from_merged_config(
            region=region,
            users=users,
            merged_config=merged_config_dict,
            config_path=config_path
        )
    else:
        from tracker_utils import auto_track_from_config
        return auto_track_from_config(
            region=region,
            users=users,
            config_path=config_path
        )


def _print_results(success: bool, region: str) -> None:
    """Print auto-tracking results."""
    print(f"\nResult:")

    if success:
        _print_success_details(region)
    else:
        _print_failure_message()


def _print_success_details(region: str) -> None:
    """Print detailed success information."""
    print(f"   AUTO-TRACK SUCCESSFUL!")
    print(f"   ")
    print(f"   The following file was modified:")

    try:
        from tracker_utils import parse_region, get_daemon_state_path
        import json

        workspace, cluster = parse_region(region)
        daemon_path = get_daemon_state_path(workspace, cluster)
        auto_track_file = daemon_path / "auto_track_requests.txt"
        print(f"File: {auto_track_file}")

        if auto_track_file.exists():
            print(f"      File exists")
            _print_last_entry(auto_track_file)
        else:
            print(f"      File does not exist (yet)")

    except Exception as verify_err:
        print(f"      Could not verify file: {verify_err}")

    print(f"   What happens next:")
    print(f"      1. AutoTrackMonitor will process this request (~10 seconds)")
    print(f"      2. Experiment will be added to experiments.json")
    print(f"      3. You will receive a Telegram notification (~30 seconds)")


def _print_last_entry(auto_track_file) -> None:
    """Print the last entry from auto-track file."""
    import json

    with open(auto_track_file, 'r') as f:
        lines = f.readlines()
        if not lines:
            return

        last_line = lines[-1].strip()
        print(f"      Last entry:")

        try:
            data = json.loads(last_line)
            print(f"         users: {data.get('users', [])}")
            print(f"         logs_path: {data.get('logs_path', 'N/A')}")
            print(f"         timestamp: {data.get('timestamp', 'N/A')}")
            if 'yaml_path' in data:
                print(f"         yaml_path: {data.get('yaml_path')}")
            if 'script_path' in data:
                print(f"         script_path: {data.get('script_path')}")
        except:
            print(f"         {last_line[:100]}...")


def _print_failure_message() -> None:
    """Print failure message."""
    print(f"   AUTO-TRACK FAILED")
    print(f"   The experiment will NOT be tracked automatically")
    print(f"   You can add it manually later using the Telegram bot")
    print(f"   ")
    print(f"   Warning: Auto-tracking failed, continuing with submission")


def _print_error(exception: Exception) -> None:
    """Print error information."""
    import traceback

    print(f"\nAUTO-TRACK ERROR:")
    print(f"   Exception: {type(exception).__name__}")
    print(f"   Message: {exception}")
    print(f"   ")
    print(f"   Warning: Auto-tracking error, continuing with submission")

    print(f"\nDetailed traceback:")
    print("-" * 80)
    traceback.print_exc()
    print("-" * 80)
