import subprocess
import os
import sys
import yaml
import socket

class ICCLLauncher:
    def __init__(self, manual_config_path):
        self.config = None
        # Priority: 1. Argument, 2. Current Dir, 3. examples/ folder
        search_paths = [
            manual_config_path,
            "./cluster.yaml",
            os.path.join(os.path.dirname(__file__), "../examples/cluster.yaml")
        ]
        
        for path in search_paths:
            if path and os.path.exists(path):
                with open(path, 'r') as f:
                    self.config = yaml.safe_load(f)
                break
        
        if not self.config:
            print("Error: cluster.yaml not found. Please create one in examples/")
            sys.exit(1)

    def _is_local(self, ip):
        """Check if the IP address belongs to the local machine."""
        try:
            # Get all local IP addresses
            local_ips = socket.gethostbyname_ex(socket.gethostname())[2]
            # Also include explicit localhost loopback
            local_ips += ["127.0.0.1", "localhost"]
            return ip in local_ips
        except:
            return False

    def orchestrate_build(self):
        common_dir = self.config['common_dir']
        built_archs = set()
        for node in self.config['nodes']:
            arch = node['type']
            if arch not in built_archs:
                print(f"[*] Orchestrating {arch} build on {node['ip']}...")
                build_dir = os.path.join(common_dir, "build", arch)
                flags = node.get('cmake_flags', "")
                
                remote_cmd = (
                    f"mkdir -p {build_dir} && cd {build_dir} && "
                    f"cmake {flags} ../.. && make -j$(nproc)"
                )

                if self._is_local(node['ip']):
                    print(f"    [Local] Detected local node. Running directly...")
                    # Run via shell=True to handle the cd and && logic
                    subprocess.run(remote_cmd, shell=True, check=True)
                else:
                    print(f"    [Remote] Dispatching via SSH...")
                    ssh_wrapper = f"bash -l -c '{remote_cmd}'"
                    subprocess.run(["ssh", "-o", "BatchMode=yes", f"root@{node['ip']}", ssh_wrapper], check=True)
                
                built_archs.add(arch)

    def launch(self, backend_name, executable, args):
        from backends.ompi import OmpiBackend
        backend = OmpiBackend() if backend_name == "ompi" else None
        if not backend: return
        
        cmd = backend.get_launch_command(self.config, executable, args)
        subprocess.run(cmd)
        