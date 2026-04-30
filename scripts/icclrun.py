#!/usr/bin/env python3
import argparse
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_DIR)

# Add the installed private library path (for production runs).
# `CMAKE_INSTALL_PREFIX/bin` -> `CMAKE_INSTALL_PREFIX/lib/infiniccl`
INSTALL_LIB_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "lib", "infiniccl")
if os.path.exists(INSTALL_LIB_PATH):
    sys.path.append(INSTALL_LIB_PATH)

from icclrun_logic import ICCLLauncher


def main():
    parser = argparse.ArgumentParser(description="InfiniCCL Unified Launcher")
    parser.add_argument("--config", "-c", dest="cluster", help="Path to cluster.yaml")
    parser.add_argument("--build", action="store_true", help="Compile remote nodes")

    launcher_args, remaining = parser.parse_known_args()

    if not remaining:
        print("Error: No executable specified.")
        sys.exit(1)

    # The first 'remaining' item is our binary, the rest are its arguments.
    executable = remaining[0]
    app_args = remaining[1:]

    launcher = ICCLLauncher(launcher_args.cluster)

    if launcher_args.build:
        launcher.orchestrate_build()

    launcher.launch("ompi", executable, app_args, launcher)


if __name__ == "__main__":
    main()
