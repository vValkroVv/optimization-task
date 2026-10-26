from git import Repo


def make_run_name_suffix(repo_path: str) -> str:
    repo = Repo(repo_path)
    last_commit = next(repo.iter_commits())
    commit_sha = str(last_commit)[:8]
    author = last_commit.author
    return f"{commit_sha} by {author}"
