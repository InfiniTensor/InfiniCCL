import pathlib
import subprocess
import sys
import tempfile


def main():
    project_root = pathlib.Path(sys.argv[1]).resolve()
    generator = project_root / "scripts" / "gen_bridge.py"

    with tempfile.TemporaryDirectory() as output_dir:
        subprocess.run(
            [
                sys.executable,
                str(generator),
                str(project_root),
                output_dir,
                "cpu;metax",
                "mccl",
            ],
            check=True,
        )

        output = pathlib.Path(output_dir)
        manifest = (output / "backend_manifest.h").read_text(encoding="utf-8")
        bridge = (output / "comm_bridge.cc").read_text(encoding="utf-8")

    expected_headers = {
        '#include "backends/ccl/mccl/metax/api.h"',
        '#include "backends/ccl/mccl/type_map.h"',
        '#include "backends/ccl/mccl/impl/send.h"',
        '#include "backends/ccl/mccl/impl/recv.h"',
    }
    for header in expected_headers:
        assert header in manifest

    send = bridge.index("infinicclResult_t infinicclSend(")
    recv = bridge.index("infinicclResult_t infinicclRecv(")
    assert "Operation<Send>::Call" in bridge[send:recv]
    assert "ReturnStatus::kNotSupported" not in bridge[send:recv]

    recv_body = bridge[recv:]
    assert "Operation<Recv>::Call" in recv_body
    assert "ReturnStatus::kNotSupported" not in recv_body.split("}", 1)[0]


if __name__ == "__main__":
    main()
