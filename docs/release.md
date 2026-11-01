# Releasing codercommittee to PyPI

Releases are built and published by the `release` workflow (`.github/workflows/release.yml`) when a
tag `vX.Y.Z` is pushed. Publishing uses PyPI's trusted publishing (OpenID Connect from GitHub
Actions): no API token is stored anywhere.

## One-time setup (the project owner, once)

1. On PyPI, sign in and open *Your projects* → *Publishing* (https://pypi.org/manage/account/publishing/).
2. Add a *pending publisher* for a new project:
   - PyPI project name: `codercommittee`
   - Owner: `codercommittee`
   - Repository name: `aicommittee`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. On GitHub, in the repository's *Settings → Environments*, create an environment named `pypi`
   (optionally with required reviewers, so a release waits for a click).

The first successful run of the workflow creates the project on PyPI and binds it to this repository.

## Every release

1. Make sure `main` is green (every CI job, including `package`).
2. Set the version in `pyproject.toml` and `src/codercommittee/__init__.py` (they must agree; the
   guard test checks the installed version), add the entry to `CHANGELOG.md`, commit, push.
3. Tag and push the tag:

   ```
   git tag -a v0.2.4 -m "codercommittee 0.2.4"
   git push origin v0.2.4
   ```

4. The workflow builds the sdist and wheel, runs `twine check`, and publishes to PyPI from the
   `pypi` environment. Check https://pypi.org/project/codercommittee/ and the Actions log.
5. Install from PyPI into a clean environment and run the quickstart once:

   ```
   uv venv /tmp/cc && uv pip install --python /tmp/cc/bin/python codercommittee
   /tmp/cc/bin/codercommittee tasks
   ```

## Dry run without publishing

`python -m pytest -m release tests/test_packaging.py` builds the distributions, checks them with
twine, installs the wheel into a fresh environment and runs the command line and the quickstart
there. CI's `package` job does the same on every push.

A test release on TestPyPI is possible by adding a second pending publisher at
https://test.pypi.org/manage/account/publishing/ with environment `testpypi` and running the
workflow with `repository-url: https://test.pypi.org/legacy/`; it is not part of the normal path.
