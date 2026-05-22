import os


class BaseMpiBackend:
    def get_hostfile_line(self, ip, slots):
        """Must return the string format for a `hostfile` line."""
        raise NotImplementedError

    def get_base_mpi_args(self, hostfile_path, total_slots):
        """Must return the initial `mpirun` execution arguments list."""
        raise NotImplementedError

    def get_env_args(self, env_key, env_val):
        """Must return the flags used to forward environment variables."""
        raise NotImplementedError

    def get_launch_command(self, config, executable, user_args, launcher_obj):
        common_dir = config["common_dir"]
        build_dir = os.path.join(common_dir, "build")
        os.makedirs(build_dir, exist_ok=True)

        # Shared Hostfile Generation
        hostfile_name = (
            f"hosts_{self.__class__.__name__.lower().replace('backend', '')}.txt"
        )
        hostfile_path = os.path.join(build_dir, hostfile_name)
        total_slots = 0

        with open(hostfile_path, "w") as f:
            for node in config["nodes"]:
                slots = node.get("slots", 8)
                total_slots += slots
                f.write(self.get_hostfile_line(node["ip"], slots))

        # Shared Launcher Script Logic
        launcher_script = config.get("launcher_script")
        if not launcher_script:
            launcher_script = launcher_obj.ensure_launcher_exists()

        # Fetch specific base runner flags (OMPI vs MPICH).
        cmd = self.get_base_mpi_args(hostfile_path, total_slots)

        # Shared Backend Arguments Parser Loop
        if "backend_args" in config and config["backend_args"]:
            for flag, values in config["backend_args"].items():
                if isinstance(values, list):
                    for val in values:
                        cmd.append(flag)
                        cmd.extend(val.split())
                else:
                    cmd.append(flag)
                    cmd.extend(str(values).split())

        # Environment Forwarding Loop
        if "backend_env" in config and config["backend_env"]:
            for env_key, env_val in config["backend_env"].items():
                cmd.extend(self.get_env_args(env_key, env_val))

        cmd.extend([launcher_script, executable])
        return cmd + list(user_args)
