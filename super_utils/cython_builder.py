"""
Cython project builder with optimization integration.

Discovers Cython projects, applies optimized compiler flags, and builds extensions.
Supports setup.py, pyproject.toml, and standalone .pyx files.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import subprocess
import glob
import os
import time
import shutil
import json


def discover_cython_project(path: str) -> Dict[str, Any]:
    """
    Detect project configuration and Cython files.

    Args:
        path: Project root directory

    Returns:
        Dict with keys:
        - pyx_files: List of .pyx file paths
        - config_type: 'setup.py' | 'pyproject.toml' | 'none'
        - config_path: Full path to config file or None
        - project_root: Absolute path to project root
    """
    root = Path(path).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if not root.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # Find configuration files
    setup_py = root / "setup.py"
    pyproject_toml = root / "pyproject.toml"

    config_type = "none"
    config_path = None

    if setup_py.exists():
        config_type = "setup.py"
        config_path = str(setup_py)
    elif pyproject_toml.exists():
        config_type = "pyproject.toml"
        config_path = str(pyproject_toml)

    # Find .pyx files (recursively)
    pyx_files = []
    for pyx_path in root.rglob("*.pyx"):
        pyx_files.append(str(pyx_path.relative_to(root)))

    # Sort for consistent ordering
    pyx_files.sort()

    return {
        'pyx_files': pyx_files,
        'config_type': config_type,
        'config_path': config_path,
        'project_root': str(root),
    }


def clean_build_artifacts(path: str) -> List[str]:
    """
    Remove Cython build artifacts.

    Removes:
    - *.so files (compiled extensions)
    - *.c files (Cython-generated, only if .pyx exists)
    - build/ directory
    - *.egg-info directories

    Args:
        path: Project root directory

    Returns:
        List of removed file/directory paths
    """
    root = Path(path).resolve()
    removed = []

    # Find .pyx files to determine which .c files are generated
    pyx_files = {p.stem for p in root.rglob("*.pyx")}

    # Remove .so files
    for so_file in root.rglob("*.so"):
        so_file.unlink()
        removed.append(str(so_file))

    # Remove generated .c files (only if corresponding .pyx exists)
    for c_file in root.rglob("*.c"):
        if c_file.stem in pyx_files:
            c_file.unlink()
            removed.append(str(c_file))

    # Remove build directory
    build_dir = root / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
        removed.append(str(build_dir))

    # Remove .egg-info directories
    for egg_info in root.rglob("*.egg-info"):
        if egg_info.is_dir():
            shutil.rmtree(egg_info)
            removed.append(str(egg_info))

    return removed


def build_cython_extensions(
    path: str,
    profile: str = "conservative",
    jobs: Optional[int] = None,
    dry_run: bool = False,
    clean: bool = False
) -> Dict[str, Any]:
    """
    Build Cython extensions with optimized flags.

    Args:
        path: Project root directory
        profile: Optimization profile ("conservative" or "aggressive")
        jobs: Parallel compilation jobs (None = CPU count)
        dry_run: Show build plan without executing
        clean: Remove artifacts before building

    Returns:
        Dict with keys:
        - success: bool
        - files_built: List of compiled .so paths
        - flags_applied: List of compiler flags used
        - build_time_seconds: float (0 if dry_run)
        - errors: List of error messages
        - dry_run: bool
        - cleaned: List of removed files (if clean=True)
    """
    from .system_spec import get_system_spec
    from .cython_optimizer import get_optimal_compile_args

    # Discover project
    try:
        project = discover_cython_project(path)
    except Exception as e:
        return {
            'success': False,
            'files_built': [],
            'flags_applied': [],
            'build_time_seconds': 0.0,
            'errors': [f"Project discovery failed: {e}"],
            'dry_run': dry_run,
        }

    if not project['pyx_files']:
        return {
            'success': False,
            'files_built': [],
            'flags_applied': [],
            'build_time_seconds': 0.0,
            'errors': ['No .pyx files found in project'],
            'dry_run': dry_run,
        }

    # Get optimization flags
    spec = get_system_spec()
    opts = get_optimal_compile_args(spec=spec, profile=profile)
    flags = opts.get('extra_compile_args', [])

    # Clean if requested
    cleaned = []
    if clean and not dry_run:
        cleaned = clean_build_artifacts(path)

    # Build result
    result = {
        'success': False,
        'files_built': [],
        'flags_applied': flags,
        'build_time_seconds': 0.0,
        'errors': [],
        'dry_run': dry_run,
    }

    if clean:
        result['cleaned'] = cleaned

    # Dry run: return plan without executing
    if dry_run:
        result['success'] = True
        result['project_info'] = project
        return result

    # Set up environment with flags
    env = os.environ.copy()
    cflags = ' '.join(flags)
    env['CFLAGS'] = cflags
    env['CXXFLAGS'] = cflags

    # Determine build command
    config_type = project['config_type']
    project_root = project['project_root']

    build_cmd = None

    if config_type == 'setup.py':
        build_cmd = ['python', 'setup.py', 'build_ext', '--inplace']
        if jobs is not None and jobs > 1:
            build_cmd.extend(['-j', str(jobs)])

    elif config_type == 'pyproject.toml':
        build_cmd = ['pip', 'install', '-e', '.', '--no-build-isolation']

    else:
        # No config: use cythonize directly
        result['errors'].append(
            "No setup.py or pyproject.toml found. "
            "Direct .pyx compilation not yet implemented. "
            "Please create a setup.py or pyproject.toml."
        )
        return result

    # Execute build
    start_time = time.time()

    try:
        proc = subprocess.run(
            build_cmd,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        build_time = time.time() - start_time
        result['build_time_seconds'] = build_time

        if proc.returncode == 0:
            result['success'] = True

            # Find compiled .so files
            root = Path(project_root)
            so_files = [str(p) for p in root.rglob("*.so")]
            result['files_built'] = so_files

            # Write build profile marker for benchmark reporting
            marker_path = root / '.superutils_build_profile.json'
            marker_data = {
                'profile': profile,
                'flags': flags,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
            with open(marker_path, 'w') as f:
                json.dump(marker_data, f, indent=2)
        else:
            result['success'] = False
            result['errors'].append(f"Build failed with exit code {proc.returncode}")
            if proc.stderr:
                result['errors'].append(f"stderr: {proc.stderr}")

    except subprocess.TimeoutExpired:
        result['errors'].append("Build timed out after 10 minutes")
    except Exception as e:
        result['errors'].append(f"Build execution failed: {e}")

    return result
