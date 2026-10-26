import os
import shutil
import sys

interval: int = 60 * 30
_MAX_LOCAL_RANKS: int = 8


SHARED_CACHE_DIR_MODE: int = 0o777


def _copy_overwriting(src: str, dst: str) -> str:
    """Copy ``src`` onto ``dst``, replacing a file we may not be able to write."""
    try:
        os.unlink(dst)
    except FileNotFoundError:
        pass
    return shutil.copy(src, dst)


def _make_dir(dst: str, *, dirs_exist_ok: bool, dir_mode: int | None) -> None:
    """Create one destination directory of the tree, optionally relaxing its mode."""
    os.makedirs(dst, exist_ok=dirs_exist_ok)
    if dir_mode is None:
        return
    try:
        os.chmod(dst, dir_mode)
    except OSError as e:
        print(f"Failed to set mode {dir_mode:o} on {dst}: {e}")


def _copytree_multithreaded(
    src,
    dst,
    symlinks=False,
    ignore=None,
    copy_function=_copy_overwriting,
    ignore_dangling_symlinks=False,
    dirs_exist_ok=False,
    max_workers=8,
    dir_mode=None,
):
    """Recursively copy a directory tree using multiple threads.

    ``dir_mode`` chmods every destination directory the copy walks; pass
    :data:`SHARED_CACHE_DIR_MODE` when publishing into a cache other users
    publish into too, and leave it ``None`` for a private destination.
    """
    sys.audit("shutil.copytree", src, dst)
    entries = list(os.scandir(src))
    _make_dir(dst, dirs_exist_ok=dirs_exist_ok, dir_mode=dir_mode)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for srcentry in entries:
            srcname = os.path.join(src, srcentry.name)
            dstname = os.path.join(dst, srcentry.name)
            if srcentry.is_dir():
                futures.append(
                    executor.submit(
                        _copytree_multithreaded,
                        srcname,
                        dstname,
                        symlinks=symlinks,
                        ignore=ignore,
                        copy_function=copy_function,
                        ignore_dangling_symlinks=ignore_dangling_symlinks,
                        dirs_exist_ok=dirs_exist_ok,
                        dir_mode=dir_mode,
                    )
                )
            else:
                futures.append(executor.submit(copy_function, srcname, dstname))

        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Failed to copy: {e}")
                raise

    return dst


def _deepgemm_cache_manager(local_rank: int) -> None:
    global _MAX_LOCAL_RANKS

    propagate_cache = os.environ.get("PROPAGATE_DG_CACHE", "")

    if not local_rank:
        if propagate_cache:
            dg_cache_dir = os.environ.get("PROPAGATE_DG_CACHE_DIR", None)
            assert dg_cache_dir and os.path.exists(dg_cache_dir), (
                f"PROPAGATE_CACHE is set but DG_CACHE_DIR={dg_cache_dir!r} "
                + "does not exist or is not set. "
                + "Check /torchtitan/ops/warmup/README.md (DeepGEMM section) "
                + "for more information."
            )
            for r in range(_MAX_LOCAL_RANKS):
                dst = f"/temp/.cache/rank_{r}/deepgemm"
                _copytree_multithreaded(dg_cache_dir, dst, dirs_exist_ok=True)
                n = sum(len(files) for _, _, files in os.walk(dst))
                print(
                    f"[cache][rank={r}] DeepGEMM: transferred {n} file(s) from {dg_cache_dir} to {dst}"
                )

        else:
            missing = [
                f"/temp/.cache/rank_{r}/deepgemm/cache"
                for r in range(_MAX_LOCAL_RANKS)
                if not os.path.exists(f"/temp/.cache/rank_{r}/deepgemm/cache")
            ]
            assert not missing, (
                "DeepGEMM cache directories missing: "
                + ", ".join(missing)
                + ". "
                + "Either set PROPAGATE_CACHE=1 with DG_CACHE_DIR or run warmup first. "
                + "Check /torchtitan/ops/warmup/README.md (DeepGEMM section) "
                + "for more information."
            )

    deepgemm_cache_dir = f"/temp/.cache/rank_{local_rank}/deepgemm/"
    # os.makedirs(deepgemm_cache_dir, exist_ok=True)
    # assert os.path.isdir(deepgemm_cache_dir)

    os.environ["DG_JIT_CACHE_DIR"] = deepgemm_cache_dir
    n_deepgemm_files = sum(len(files) for _, _, files in os.walk(deepgemm_cache_dir))
    print(f"DeepGEMM: {n_deepgemm_files=} from {deepgemm_cache_dir}")
