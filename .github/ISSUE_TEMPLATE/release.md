---
name: Release checklist
about: '[for maintainer use]'
title: 'Release freud 3.5.0'
labels: ''

---

To make a new freud release:

- [ ] Make a new `release-{X.Y.Z}` branch (where `{X.Y.Z}` is the new version).

On that branch, take the following steps (committing after each step when needed):

- [ ] Run `prek autoupdate`.
- [ ] Check for new or duplicate contributors since the last release:
  `comm -13 (git log $(git describe --tags --abbrev=0) --format="%aN <%aE>" | sort | uniq | psub) (git log --format="%aN <%aE>" | sort | uniq | psub)`.
  Add entries to `.mailmap` to remove duplicates.
- [ ] Review the change log and revise if needed.
- [ ] Run `bump-my-version bump {type}`. Replace `{type}` with:
  - `patch` when this release *only* includes bug fixes.
  - `minor` when this release includes new features and possibly bug fixes.
  - `major` when this release includes API breaking changes.
- [ ] Push the branch and open a pull request.
- [ ] Check that readthedocs built the docs correctly in the pull request checks.
- [ ] Merge the pull request after all tests pass.
- [ ] Make a new tag on the main branch:
  ```
  git switch main
  git pull
  git tag -a v{X.Y.Z}
  git push origin --tags
  ```
  Make sure to include the `v` in the tag name!
- [ ] Add a blank changelog entry for the next release:
  ```
  ## Next release

  ###  Added

  ### Changed

  ### Deprecated

  ### Removed
  
  ### Fixed
  ```

GitHub Actions will trigger on the tag and upload new wheels to PyPI and create a
GitHub release. After a few hours, the conda-forge autotick bot will submit a PR
for the new release.

- [ ] Check that the GitHub release posted correctly.
- [ ] Check that the PyPI wheels all uploaded correctly.
- [ ] Merge the conda-forge recipe, updating it first if necessary (e.g. when adding dependencies).
