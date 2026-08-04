# Releasing

Protocol releases use tags named `protocol-v<version>` in the
`pengyue-polaron/galaxea-a1-runtime` repository. The `publish-protocol.yml`
workflow builds this package directory and publishes through the
`pypi-protocol` GitHub environment. The PyPI Trusted Publisher must name that
repository, workflow, and environment exactly.
