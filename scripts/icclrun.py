#!/usr/bin/env python3
import argparse
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_DIR)

from icclrun_logic import ICCLLauncher

def main():
    parser = argparse.ArgumentParser(description="InfiniCCL Unified Launcher")
    parser.add_argument("--cluster", help="Path to cluster.yaml")
    parser.add_argument("--build", action="store_true", help="Compile remote nodes")
    parser.add_argument("executable", help="Binary name")
    parser.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    launcher = ICCLLauncher(args.cluster)
    
    if args.build:
        launcher.orchestrate_build()
    
    launcher.launch("ompi", args.executable, args.args)

if __name__ == "__main__":
    main()
    