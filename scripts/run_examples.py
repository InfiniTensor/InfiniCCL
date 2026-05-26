#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional


# ==============================================================================
# ENGINE RUNNER
# ==============================================================================
def setup_log_directory(directory: str) -> str:
    """Creates a unique timestamped subdirectory inside the log folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = os.path.join(directory, f"run_{timestamp}")
    os.makedirs(target_path, exist_ok=True)
    return target_path


def run_iccl_example(
    example_name: str,
    config_path: str,
    launcher_opt: Optional[str],
    log_dir: str,
    trigger_build: bool,
    verbose: bool,
    timeout_duration: int,
) -> bool:
    """Executes an example via `icclrun` orchestration framework."""
    log_file_path = os.path.join(log_dir, f"{example_name}.log")

    status_msg = (
        "🚀 Running via `icclrun` (with build):"
        if trigger_build
        else "🚀 Running via `icclrun`:             "
    )
    print(f"{status_msg:<35} {example_name:<20}", end="", flush=True)

    # Base Command Assembly
    cmd = ["icclrun", "--config", config_path]
    if launcher_opt and launcher_opt.strip():
        cmd.extend(["--launcher", launcher_opt.strip()])

    if trigger_build:
        cmd.extend(["--build", example_name])
    else:
        cmd.append(example_name)

    # Format the exact command as a clean string for log documentation.
    exact_command_str = " ".join(cmd)

    # Force environment unbuffered stream states for sub-python instances.
    custom_env = os.environ.copy()
    custom_env["PYTHONUNBUFFERED"] = "1"

    try:
        with open(log_file_path, "w", buffering=1) as log_file:
            # Write the exact underlying command as the absolute first line of the log.
            log_file.write(f"[COMMAND]: {exact_command_str}\n")
            log_file.write("=" * 80 + "\n\n")
            log_file.flush()

            if verbose:
                print(f"\n--- [VERBOSE OUTPUT START: `{example_name}`] ---")

                # Execute with `Popen` to stream `stdout`/`stderr` live to both terminal and file.
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=custom_env,
                    text=True,
                )

                for line in process.stdout:
                    sys.stdout.write(line)
                    log_file.write(line)

                process.wait(timeout=timeout_duration)
                return_code = process.returncode
                print(
                    f"--- [VERBOSE OUTPUT END: `{example_name}`] ---\n" + " " * 56,
                    end="",
                )
            else:
                # Quiet mode: Redirect straight to the file handle.
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=custom_env,
                    text=True,
                    timeout=timeout_duration,
                )
                return_code = result.returncode

        if return_code == 0:
            print(f"  ✓ PASSED  (Log saved to `{os.path.basename(log_file_path)}`)")
            return True
        else:
            print(f" ❌ FAILED  (Exit Code: {return_code})")
            return False

    except subprocess.TimeoutExpired:
        print(f" ❌ TIMEOUT (Exceeded {timeout_duration} seconds)")
        with open(log_file_path, "a") as f:
            f.write(
                f"\n[RUNNER ERROR]: Distributed `icclrun` harness timed out after {timeout_duration} seconds.\n"
            )
        return False
    except FileNotFoundError:
        print(" ❌ ERROR   (`icclrun` executable not found in `PATH`)")
        return False
    except Exception as e:
        print(f" ❌ CRASHED ({str(e)})")
        return False


def main():
    # Setup CLI Argument Parsing
    parser = argparse.ArgumentParser(
        description="InfiniCCL Example Suite Execution Harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="examples/cluster.yaml",
        help="Path to the cluster topology YAML configuration file (i.e., `cluster.yaml`).",
    )
    parser.add_argument(
        "-l",
        "--launcher",
        type=str,
        default=None,
        help="Specify the target launcher. If omitted, defaults to `icclrun`'s default. See `icclrun --help` for valid options.",
    )
    parser.add_argument(
        "-e",
        "--examples",
        type=str,
        default="all_reduce",
        help="Comma-separated list of example target names to execute.",
    )
    parser.add_argument(
        "-o",
        "--log-path",
        type=str,
        default="./example_logs",
        help="Destination directory where output logs will be stored.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=300,
        help="Timeout duration in seconds for each individual example execution pass.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose mode: stream execution `stdout`/`stderr` directly to terminal while logging.",
    )

    args = parser.parse_args()

    # Parse and clean the comma-separated examples list.
    examples_to_run = [ex.strip() for ex in args.examples.split(",") if ex.strip()]

    # Sanity Checks
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file matching '{args.config}' could not be located.")
        sys.exit(1)

    if not examples_to_run:
        print(
            "❌ Error: No target programs defined. Please check your `--examples` list."
        )
        sys.exit(1)

    # Initialize Log Infrastructure
    current_run_log_dir = setup_log_directory(args.log_path)
    launcher_display = args.launcher if args.launcher else "Internal default"

    print("==================================================================")
    print("               InfiniCCL Distributed Verification                 ")
    print(f"Target Configuration File: {args.config}")
    print(f"Selected Launcher Engine : {launcher_display}")
    print(f"Logs Target Destination  : {current_run_log_dir}")
    print(f"Verbose Console Output   : {'ENABLED' if args.verbose else 'DISABLED'}")
    print("==================================================================")

    passed_count = 0
    failed_count = 0
    failed_programs = []
    is_first_run = True

    for example in examples_to_run:
        success = run_iccl_example(
            example_name=example,
            config_path=args.config,
            launcher_opt=args.launcher,
            log_dir=current_run_log_dir,
            trigger_build=is_first_run,
            verbose=args.verbose,
            timeout_duration=args.timeout,
        )

        is_first_run = False

        if success:
            passed_count += 1
        else:
            failed_count += 1
            failed_programs.append(example)

    # ==============================================================================
    # METRICS REPORTING
    # ==============================================================================
    total_programs = len(examples_to_run)
    success_rate = (passed_count / total_programs) * 100

    print("\n==================================================================")
    print("                      EXECUTION SUMMARY                           ")
    print("==================================================================")
    print(f"Total Framework Targets  : {total_programs}")
    print(f"Successfully Passed      : {passed_count}")
    print(f"Failed / Crashed Targets : {failed_count}")
    print(f"Overall Success Rate     : {success_rate:.2f}%")
    print("==================================================================")

    if failed_count > 0:
        print(
            f"⚠️  The following collective validation targets failed: {', '.join(failed_programs)}"
        )
        sys.exit(1)
    else:
        print("🎉 All targets executed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
