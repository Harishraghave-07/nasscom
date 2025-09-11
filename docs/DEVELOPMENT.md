Development & reproducible-build notes

This project ships a lightweight Option A developer workflow: a draft Dockerfile
and an unlocked `requirements.in` so you can produce a pinned lockfile locally.

Generate a deterministic lockfile (recommended)

1) Install pip-tools in your dev env:

   python -m pip install --user pip-tools

2) Generate a pinned lockfile from `requirements.in`:

   pip-compile requirements.in --output-file=requirements.lock

3) Use `requirements.lock` for deterministic installs or in CI/Docker:

   pip install -r requirements.lock

Notes about heavy dependencies
- Heavy libraries (torch, easyocr, presidio) are intentionally left commented
  in `requirements.in`. Add and pin them in `requirements.in`/`pyproject.toml`
  before running `pip-compile` to bake exact versions into `requirements.lock`.

Docker build (draft CPU-only image)

# build
docker build -t nasscom:dev .

# run (quick smoke): runs pytest inside the container
docker run --rm nasscom:dev

CI guidance
- Add the generated `requirements.lock` to the repository for deterministic CI
  builds.
- The repo includes `.github/workflows/ci-lockfile-lint.yml` which enforces the
  presence of `requirements.lock` and checks for absolute user paths. Adjust as
  needed for your CI policy.
