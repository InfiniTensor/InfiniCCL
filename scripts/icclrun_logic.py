import os
import socket
import subprocess
import sys
from pathlib import Path

import yaml


class ICCLLauncher:
    def __init__(self, manual_config_path):
        self.config = None
        # Priority: 1. Argument, 2. Current Dir, 3. examples/ folder.
        search_paths = [
            manual_config_path,
            "./cluster.yaml",
            os.path.join(os.path.dirname(__file__), "../examples/cluster.yaml"),
        ]

        for path in search_paths:
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    self.config = yaml.safe_load(f)
                break

        if not self.config:
            print("Error: cluster.yaml not found.")
            sys.exit(1)

        # Detect InfiniCCL Root
        self.infiniccl_root = os.environ.get("INFINICCL_ROOT")
        if not self.infiniccl_root:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            self.infiniccl_root = os.path.abspath(os.path.join(script_dir, "../../.."))

    def _is_local(self, ip):
        """Check if the IP address belongs to the local machine."""
        try:
            local_ips = socket.gethostbyname_ex(socket.gethostname())[2]
            local_ips += ["127.0.0.1", "localhost"]
            return ip in local_ips
        except:
            return False

    def orchestrate_build(self):
        common_dir = self.config["common_dir"]
        infiniccl_root = self.infiniccl_root
        is_internal = os.path.abspath(common_dir) == os.path.abspath(infiniccl_root)
        base_install = self.config.get("install_dir", common_dir)

        for node in self.config["nodes"]:
            arch = node["type"]
            install_path = os.path.join(base_install, "install", arch)
            user_cmake_flags = node.get("cmake_flags", "")

            # Build the library using YAML flags.
            lib_cmd = (
                f"mkdir -p {infiniccl_root}/build/{arch} && cd {infiniccl_root}/build/{arch} && "
                f"cmake -DCMAKE_INSTALL_PREFIX={install_path} {user_cmake_flags} {infiniccl_root} && "
                f"make -j$(nproc) install"
            )

            if is_internal:
                full_cmd = lib_cmd
            else:
                app_cmd = (
                    f"export INFINICCL_INSTALL={install_path} && "
                    f"mkdir -p {common_dir}/build/{arch} && cd {common_dir}/build/{arch} && "
                    f"cmake {user_cmake_flags} {common_dir} && "
                    f"make -j$(nproc)"
                )
                full_cmd = f"{lib_cmd} && {app_cmd}"

            # Execute via SSH or locally.
            user = node.get("user", self.config.get("common_user", "root"))
            exec_cmd = f"bash -l -c '{full_cmd}'"
            print(f"[*] Orchestrating {arch} on {node['ip']}...")
            if self._is_local(node["ip"]):
                subprocess.run(exec_cmd, shell=True, check=True)
            else:
                subprocess.run(["ssh", f"{user}@{node['ip']}", exec_cmd], check=True)

        return self.ensure_launcher_exists()

    def ensure_launcher_exists(self):
        # This must be an absolute path.
        wrapper_path = os.path.abspath(
            os.path.join(self.config["common_dir"], "build", "run_wrapper.sh")
        )
        os.makedirs(os.path.dirname(wrapper_path), exist_ok=True)

        common_dir = Path(self.config["common_dir"]).expanduser().resolve()
        infiniccl_root_dir = Path(self.infiniccl_root).expanduser().resolve()
        is_internal = common_dir == infiniccl_root_dir

        bin_sub = "examples/$1" if is_internal else "$1"

        case_blocks = []
        first = True

        for node in self.config["nodes"]:
            n_type = node["type"]
            n_env = node.get("backend_env", {})

            # Generic environment injection from YAML.
            exports = f'    export LD_LIBRARY_PATH="{self.infiniccl_root}/install/{n_type}/lib:${{LD_LIBRARY_PATH}}"\n'

            for k, v in n_env.items():
                if k == "LD_LIBRARY_PATH":
                    v = f"{v}:${{LD_LIBRARY_PATH}}"
                exports += f'    export {k}="{v}"\n'

            condition = None

            if n_type == "nvidia":
                condition = '[ -c "/dev/nvidia0" ] || [ -x "$(command -v nvidia-smi)" ]'

            elif n_type == "metax":
                condition = (
                    '[ -d "/opt/maca" ] || '
                    'grep -l "9999" /sys/bus/pci/devices/*/vendor >/dev/null 2>&1'
                )

            # Skip unknown platforms cleanly.
            if condition is None:
                continue

            keyword = "if" if first else "elif"
            first = False

            case_blocks.append(
                f'{keyword} {condition}; then\n{exports}    ARCH="{n_type}"\n'
            )

        # Fallback when no accelerator matched (or no platforms configured).
        if case_blocks:
            case_blocks.append('else\n    ARCH="cpu"\nfi\n')
        else:
            case_blocks.append('ARCH="cpu"\n')

        content = f"""#!/bin/bash
{"".join(case_blocks)}
EXE="{self.config["common_dir"]}/build/$ARCH/{bin_sub}"
shift
exec "$EXE" "$@"
    """

        with open(wrapper_path, "w") as f:
            f.write(content)

        os.chmod(wrapper_path, 0o755)
        return wrapper_path

    def launch(self, backend_name, executable, args, launcher):
        from backends.ompi import OmpiBackend

        backend = OmpiBackend() if backend_name == "ompi" else None
        if not backend:
            return

        cmd = backend.get_launch_command(self.config, executable, args, launcher)
        subprocess.run(cmd)
