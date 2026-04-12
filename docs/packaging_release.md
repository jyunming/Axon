# Packaging & Release (Python + Rust)

## User install
Users still install a single package:

```bash
pip install axon-rag
```

If a platform wheel with native extension is available, `axon_rust` is installed automatically.
If not, Axon falls back to pure Python at runtime.

## Single-source versioning
Authoritative version location:

- `src/axon/Cargo.toml` -> `[package].version`

`pyproject.toml` uses `project.dynamic = ["version"]`, so wheel metadata is derived from Cargo.

## Bump version

```bash
python scripts/bump_version.py 0.2.0
python scripts/audit_packaging.py --expected-version 0.2.0
```

## Release by tag
Push a tag matching Cargo version:

```bash
git tag v0.2.0
git push origin v0.2.0
```

Release workflow builds:
- wheels: Linux, macOS, Windows
- sdist

Then uploads artifacts to GitHub Release and PyPI.

## PyPI Trusted Publisher (OIDC)
Configure once in PyPI (Project -> Publishing -> Add publisher):

- Owner: `jyunming`
- Repository name: `studio_brain_open`
- Workflow name: `release.yml`
- Environment name: (leave empty unless you enforce one in GitHub)

After that, no `PYPI_API_TOKEN` secret is required. Publishing is performed by:

`pypa/gh-action-pypi-publish@release/v1`

with GitHub Actions permission:

`id-token: write`

## Safety checks
CI + Release run:

```bash
python scripts/audit_packaging.py
```

This enforces:
- `maturin` backend
- dynamic versioning from Cargo
- no duplicate Rust manifest
- tag version matches Cargo version (release workflow)
