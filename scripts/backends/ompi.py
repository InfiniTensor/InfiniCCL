import os


class OmpiBackend:
    def get_launch_command(self, config, executable, user_args, launcher_obj):
        common_dir = config["common_dir"]
        build_dir = os.path.join(common_dir, "build")
        os.makedirs(build_dir, exist_ok=True)

        # Hostfile Generation
        hostfile_path = os.path.join(build_dir, "hosts.txt")
        total_slots = 0
        with open(hostfile_path, "w") as f:
            for node in config["nodes"]:
                slots = node.get("slots", 8)
                total_slots += slots
                f.write(f"{node['ip']} slots={slots}\n")

        # Launcher Logic: Use YAML override OR Auto-generate.
        launcher_script = config.get("launcher_script")
        if not launcher_script:
            launcher_script = launcher_obj.ensure_launcher_exists()

        cmd = ["mpirun", "--hostfile", hostfile_path, "-np", str(total_slots)]

        # Safety: OpenMPI refuses to run as root unless explicitly told.
        if os.getuid() == 0:
            cmd.append("--allow-run-as-root")

        if "backend_args" in config and config["backend_args"]:
            for flag, values in config["backend_args"].items():
                if isinstance(values, list):
                    for val in values:
                        cmd.append(flag)
                        cmd.extend(val.split())
                else:
                    cmd.append(flag)
                    cmd.extend(str(values).split())

        if "backend_env" in config and config["backend_env"]:
            for env_key, env_val in config["backend_env"].items():
                cmd.extend(["-x", f"{env_key}={env_val}"])

        cmd.extend([launcher_script, executable])
        return cmd + list(user_args)
