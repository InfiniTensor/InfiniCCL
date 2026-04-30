import os
import re
import sys

# =================================================================
# CONFIGURATION
# =================================================================

# The standardized warning header for generated files.
AUTOGEN_HEADER = """/*
 * -----------------------------------------------------------------
 * AUTO-GENERATED FILE — DO NOT EDIT.
 * -----------------------------------------------------------------
 * This file is generated during the build process.
 * Any changes made here may be overwritten during the build or code
 * generation process.
 * -----------------------------------------------------------------
 */
"""

# Hardware traits to look for in device directories.
DEVICE_TRAIT_HEADERS = ["device_.h", "runtime_.h", "data_type_.h"]

# Map logical backend names (from CMake) to their internal source paths.
BACKEND_PATH_MAP = {"ompi": "ompi/impl", "nccl": "nvidia/nccl"}

# =================================================================
# LOGIC
# =================================================================


def snake_to_camel(name):
    """Converts infini_all_reduce or infiniAllReduce to AllReduce."""
    clean_name = re.sub(r"^infini", "", name)
    if "_" in clean_name:
        return "".join(x.capitalize() for x in clean_name.split("_"))
    return clean_name[0].upper() + clean_name[1:]


def parse_signatures(header_path):
    """Extracts function metadata from the public C header."""
    signatures = []
    regex = r"(infiniResult_t)\s+(infini\w+)\s*\((.*?)\);"
    if not os.path.exists(header_path):
        return signatures

    with open(header_path, "r") as f:
        content = f.read()
        for m in re.finditer(regex, content, re.DOTALL):
            ret_type, func_name, params_raw = m.group(1), m.group(2), m.group(3).strip()

            param_names = []
            if params_raw and params_raw != "void":
                for p in params_raw.split(","):
                    # Extract the variable name (last word before comma/bracket).
                    param_names.append(p.strip().split()[-1].replace("*", ""))

            signatures.append(
                {
                    "ret": ret_type,
                    "name": func_name,
                    "params": params_raw,
                    "args": ", ".join(param_names),
                    "key": snake_to_camel(func_name),
                }
            )
    return signatures


def generate(project_root, output_dir, devices, backends):
    src_dir = os.path.join(project_root, "src")
    base_dir = os.path.join(src_dir, "base")
    header_path = os.path.join(project_root, "include/comm.h")

    # Get the list of all operations defined in the base directory.
    ops_in_base = (
        [f[:-2] for f in os.listdir(base_dir) if f.endswith(".h")]
        if os.path.exists(base_dir)
        else []
    )
    sigs = parse_signatures(header_path)

    # 1. Generate `backend_manifest.h`.
    manifest_lines = [
        AUTOGEN_HEADER,
        "#ifndef INFINI_CCL_BACKEND_MANIFEST_H_",
        "#define INFINI_CCL_BACKEND_MANIFEST_H_",
    ]

    found_devices = []

    # Process Active Devices
    for dev in devices:
        if not dev:
            continue

        device_included = False
        manifest_lines.append(f"\n// --- DEVICE: {dev.upper()} ---")

        for trait in DEVICE_TRAIT_HEADERS:
            rel_path = f"{dev}/{trait}"
            if os.path.exists(os.path.join(src_dir, rel_path)):
                manifest_lines.append(f'#include "{rel_path}"')
                device_included = True

        if device_included:
            found_devices.append(f"Device::Type::k{dev.capitalize()}")

    # Process Active Backends
    for bb in backends:
        if not bb or bb not in BACKEND_PATH_MAP:
            continue

        manifest_lines.append(f"\n// --- BACKEND: {bb.upper()} ---")
        impl_subpath = BACKEND_PATH_MAP[bb]

        for op in ops_in_base:
            rel_path = f"{impl_subpath}/{op}.h"
            if os.path.exists(os.path.join(src_dir, rel_path)):
                manifest_lines.append(f'#include "{rel_path}"')

    # Add the Type Alias `EnabledDevices`
    manifest_lines.append("\nnamespace infini::ccl {\n")
    if found_devices:
        dev_list_str = ", ".join(found_devices)
        manifest_lines.append(f"using EnabledDevices = List<{dev_list_str}>;")
    else:
        manifest_lines.append("using EnabledDevices = List<>;")
    manifest_lines.append("\n} // namespace infini::ccl")

    manifest_lines.append("\n#endif // INFINI_CCL_BACKEND_MANIFEST_H_\n")

    with open(os.path.join(output_dir, "backend_manifest.h"), "w") as f:
        f.write("\n".join(manifest_lines))

    # 2. Generate `comm_bridge.cc`.
    bridge_lines = [
        AUTOGEN_HEADER,
        '#include "backend_manifest.h"',
        '#include "comm.h"',
        '#include "comm_impl.h"',
        '#include "data_type_impl.h"',
        '#include "operation.h"',
        "\nnamespace infini::ccl {",
        '\nextern "C" {',
    ]

    for s in sigs:
        # We need to transform the raw args to add casts for specific types.
        args_with_casts = []
        # Split params to analyze types (simplified approach).
        params_list = s["params"].split(",")

        for p in params_list:
            p = p.strip()
            if not p or p == "void":
                continue

            # Get the type and the name.
            parts = p.split()
            arg_type = parts[0]
            arg_name = parts[-1].replace("*", "")

            # Apply `static_cast` for specialized `Infini` types.
            if arg_type == "infiniDataType_t":
                args_with_casts.append(f"static_cast<DataType>({arg_name})")
            elif arg_type == "infiniRedOp_t":
                args_with_casts.append(f"static_cast<ReductionOpType>({arg_name})")
            else:
                args_with_casts.append(arg_name)

        casted_args_str = ", ".join(args_with_casts)

        bridge_lines.append(
            f"\n{s['ret']} {s['name']}({s['params']}) {{\n"
            f"    return static_cast<{s['ret']}>(Operation<{s['key']}>::Call({casted_args_str}));\n"
            f"}}"
        )

    bridge_lines.append('\n} // extern "C"')
    bridge_lines.append("} // namespace infini::ccl\n")

    with open(os.path.join(output_dir, "comm_bridge.cc"), "w") as f:
        f.write("\n".join(bridge_lines))


if __name__ == "__main__":
    # Path/List arguments passed from CMake.
    root = sys.argv[1]
    out = sys.argv[2]
    devs = sys.argv[3].split(";") if len(sys.argv) > 3 else []
    backs = sys.argv[4].split(";") if len(sys.argv) > 4 else []

    generate(root, out, devs, backs)
