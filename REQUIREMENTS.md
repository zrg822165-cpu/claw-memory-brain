# Dependency Notes

This public repository keeps the original requirement files mostly intact, but they come from a prototype environment rather than a polished release process.

## Files

- `requirements-wsl.txt`
  - lighter starting point
  - best for reading the project, exploring the main retrieval flow, or doing minimal local setup

- `requirements.txt`
  - broader experiment-oriented dependency set
  - includes packages that were useful during earlier prototype work, not necessarily for every public user

## Practical Guidance

If you want the smallest reasonable starting point, begin with:

```bash
pip install -r requirements-wsl.txt
```

Then only add more packages if your workflow actually reaches those code paths.

## Notes

- Some dependencies reflect the original WSL-first experimentation environment.
- Some graph-related packages were part of exploratory work and may not be needed for basic retrieval/routing inspection.
- The public repo should be treated as a research prototype snapshot, so a bit of manual environment adjustment is expected.
