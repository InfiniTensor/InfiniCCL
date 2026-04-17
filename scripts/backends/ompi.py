import os

class OmpiBackend:
    def get_launch_command(self, config, executable, user_args):
        common_dir = config['common_dir']
        hostfile_path = os.path.join(common_dir, "build", "hosts.txt")
        launcher_script = os.path.join(common_dir, "scripts/run_wrapper.sh")
        
        total_slots = 0
        with open(hostfile_path, "w") as f:
            for node in config['nodes']:
                slots = node.get('slots', 8)
                total_slots += slots
                f.write(f"{node['ip']} slots={slots}\n")
        
        cmd = [
            "mpirun", "--allow-run-as-root",
            "--hostfile", hostfile_path,
            "-np", str(total_slots)
        ]

        if 'backend_args' in config and config['backend_args']:
            for flag, values in config['backend_args'].items():
                if isinstance(values, list):
                    for val in values:
                        cmd.append(flag)
                        cmd.extend(val.split()) 
                else:
                    cmd.append(flag)
                    cmd.extend(str(values).split())

        if 'backend_env' in config and config['backend_env']:
            for env_key, env_val in config['backend_env'].items():
                cmd.extend(["-x", f"{env_key}={env_val}"])
        
        cmd.extend([launcher_script, executable])
        return cmd + list(user_args)
    