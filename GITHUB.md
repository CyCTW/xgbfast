# GitHub preparation

This directory is a standalone Git repository. No GitHub remote is configured
and nothing has been uploaded. Local files in the original workspace remain intact.

Before the first GitHub upload, choose the owner/repository name and visibility.
Before public open-source release, choose a project license and review the
licenses/notices for any material redistributed with the release. No license
has been selected automatically.

Upload this repository's root, not the parent workspace. CI lives in
`.github/workflows/linux.yml` and runs tests, compilation, runtime validation,
and Python benchmarks on x86-64 and ARM64. CI timing has no performance gate.

Model artifacts, virtual environments, binaries, and local benchmark outputs
are ignored. Build wheels per supported Python/OS/architecture; a local
`linux_aarch64` wheel is not a manylinux compatibility claim.

This project is not published on PyPI. GitHub upload and package publication
are separate actions.
