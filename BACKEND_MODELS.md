Repository model handling

Problem
-------
The trained model files (e.g., `backend/models/model_1.pkl`) are large (>100 MB) and exceed GitHub's file size limit. A recent `git push` failed due to these large files being present in history.

Options
-------
1) Use Git LFS (recommended if you want models inside the repo)
   - Install Git LFS and track `.pkl` files, then push.
   - Pros: models stored alongside code, familiar workflow for contributors
   - Cons: contributors must have Git LFS, storage limits on GitHub LFS (billing)

   Steps (PowerShell):
   ```powershell
   # Install git-lfs (if not installed)
   choco install git-lfs -y  # or follow https://git-lfs.github.com install instructions
   git lfs install
   git lfs track "backend/models/*.pkl"
   git add .gitattributes
   git add backend/models/*.pkl
   git commit -m "chore: add models via Git LFS"
   git push origin main
   ```

2) Host models externally and exclude from repo (recommended for large or many models)
   - Keep code in Git; host model files on cloud storage (AWS S3, Google Drive, Google Cloud Storage, or an artifact repo) and provide a small downloader script.
   - Pros: no repo size or billing issues, faster git operations
   - Cons: need to manage external hosting and access controls

   Steps:
   - Remove model files from Git history (see 'Clean history' below)
   - Add `backend/models/` to `.gitignore`
   - Provide a `scripts/download_models.py` that downloads models from your chosen host

3) Remove large files from Git history and push repository without them
   - Use `git filter-repo` or BFG Repo-Cleaner to strip `*.pkl` files from history, then force-push.
   - This permanently rewrites history; any collaborator must re-clone.

   Example using BFG (PowerShell):
   ```powershell
   # Install Java, download bfg.jar
   java -version
   # download bfg.jar from https://rtyley.github.io/bfg-repo-cleaner/
   # Make a bare clone
   git clone --mirror <local-repo-path> repo.git
   cd repo.git
   # Remove files by name pattern
   java -jar ..\bfg.jar --delete-files *.pkl
   git reflog expire --expire=now --all && git gc --prune=now --aggressive
   git push --force
   ```

Recommended approach for this project
-------------------------------------
Because model files are large and may change frequently, I recommend option (2): host models externally (S3 or Google Drive) and add a `scripts/download_models.py` to fetch them on setup. This keeps the git repo lightweight and portable.

Next steps I can do for you
---------------------------
- Add `scripts/download_models.py` with placeholders and usage instructions. (I can add this file now.)
- Add `.gitignore` entry to ignore `backend/models/*.pkl`. (I can create a PR or show exact commands.)
- If you prefer Git LFS, I can provide the exact commands and help migrate.
- If you want me to strip the `.pkl`'s from history, I can provide step-by-step commands (requires tooling and a force-push).

Tell me which option you prefer and I will implement the corresponding repository changes.
