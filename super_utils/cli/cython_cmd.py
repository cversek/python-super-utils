"""
Cython optimization CLI commands.

Commands:
- superutils cython detect     - Show hardware + recommended flags
- superutils cython recommend  - Just show recommended flags (no hardware details)
- superutils cython compile    - Build Cython extensions with optimized flags

Note: Benchmarks moved to 'superutils benchmark' command.
"""

import argparse
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from ..cython_optimizer import get_optimal_compile_args
from ..system_spec import get_system_spec, get_apple_silicon_info, get_linux_cpu_info


def add_cython_parser(subparsers):
    """Add cython subcommand parser."""
    parser = subparsers.add_parser(
        'cython',
        help='Cython compiler optimization tools',
        description='Detect hardware, benchmark optimizations, and build Cython extensions'
    )

    parser.add_argument(
        'action',
        choices=['detect', 'recommend', 'compile'],
        help='Action to perform'
    )

    # Common arguments
    parser.add_argument(
        '--profile',
        choices=['O0', 'O1', 'O2', 'O3', 'O3-march', 'O3-march-vec', 'O3-ffast-math'],
        default='O3-march',
        help='Optimization profile (default: O3-march)'
    )

    # Compile-specific arguments
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Project path for compile action (default: current directory)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show build plan without executing'
    )

    parser.add_argument(
        '--jobs',
        type=int,
        help='Parallel compilation jobs (default: CPU count)'
    )

    parser.add_argument(
        '--clean',
        action='store_true',
        help='Remove build artifacts before compiling'
    )

    return parser


def handle_cython(args):
    """Handle cython subcommand."""
    console = Console()

    if args.action == 'detect':
        return _handle_detect(console, args)
    elif args.action == 'recommend':
        return _handle_recommend(console, args)
    elif args.action == 'compile':
        return _handle_compile(console, args)
    else:
        console.print(f"[red]Unknown action: {args.action}[/red]")
        return 1


def _handle_detect(console: Console, args):
    """Handle detect action."""
    spec = get_system_spec()
    opts = get_optimal_compile_args(spec=spec, profile=args.profile)

    _show_hardware_detection(console, spec, opts)
    console.print()
    _show_recommendations(console, opts)

    return 0


def _handle_recommend(console: Console, args):
    """Handle recommend action."""
    spec = get_system_spec()
    opts = get_optimal_compile_args(spec=spec, profile=args.profile)

    _show_recommendations(console, opts)

    return 0


def _show_hardware_detection(console: Console, spec: dict, opts: dict):
    """Display hardware detection results."""
    hw = spec.get('hardware', {})
    os_info = spec.get('os', {})

    # Hardware detection table
    table = Table(title="Hardware Detection", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="white")

    table.add_row("Platform", f"{os_info.get('platform', '?')} {hw.get('architecture', '?')}")
    table.add_row("CPU Model", str(hw.get('cpu_model', 'unknown')))
    table.add_row("Cores (logical)", str(hw.get('cpu_cores_logical', '?')))

    # Platform-specific details
    platform_type = opts.get('platform')

    if platform_type == 'apple_silicon':
        chip = get_apple_silicon_info()
        if chip:
            chip_name = f"{chip.get('generation', '?')} {chip.get('variant', '').strip() or 'base'}"
            table.add_row("Apple Silicon", f"[green]{chip_name}[/green]")
            table.add_row("Performance Cores", str(chip.get('performance_cores', '?')))
            table.add_row("Efficiency Cores", str(chip.get('efficiency_cores', '?')))

    elif platform_type in ('linux_x86_64', 'linux_arm64'):
        cpu = get_linux_cpu_info()
        if cpu:
            if cpu.get('total_cores'):
                table.add_row("Physical Cores", str(cpu['total_cores']))

            features = cpu.get('features', [])
            if features:
                # Highlight important features
                important = []
                if 'avx512f' in features:
                    important.append('[green]AVX-512[/green]')
                elif 'avx2' in features:
                    important.append('[green]AVX2[/green]')
                elif 'avx' in features:
                    important.append('[yellow]AVX[/yellow]')

                if 'fma' in features:
                    important.append('[green]FMA[/green]')

                if 'neon' in [f.lower() for f in features]:
                    important.append('[green]NEON[/green]')

                if important:
                    table.add_row("SIMD Features", ", ".join(important))

    console.print(table)


def _show_recommendations(console: Console, opts: dict):
    """Display compiler flag recommendations."""
    profile = opts.get('profile', 'conservative')
    flags = opts.get('extra_compile_args', [])
    reasoning = opts.get('reasoning', {})

    # Title panel
    profile_color = "yellow" if profile == "aggressive" else "green"
    title = f"Recommended Compiler Flags ([{profile_color}]{profile.upper()}[/{profile_color}])"

    # Flags table
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Flag", style="cyan", width=25)
    table.add_column("Explanation", style="white")

    for flag in flags:
        explanation = reasoning.get(flag, "No explanation available")

        # Highlight warnings for aggressive flags
        if 'AGGRESSIVE' in explanation:
            explanation = f"[yellow]{explanation}[/yellow]"

        table.add_row(flag, explanation)

    console.print(table)

    # Warning for aggressive profile
    if profile == "aggressive":
        warning_text = (
            "[bold yellow]WARNING:[/bold yellow] Aggressive profile uses -ffast-math which may:\n"
            "  • Break code that depends on IEEE 754 NaN/Inf behavior\n"
            "  • Produce incorrect results for edge cases\n"
            "  • Make debugging harder\n\n"
            "Only use if you've profiled and understand the tradeoffs!"
        )
        console.print(Panel(warning_text, border_style="yellow", title="Aggressive Profile Warning"))

    # Usage example
    console.print()
    usage_code = f"""# In your setup.py:
from super_utils.cython_optimizer import get_optimal_compile_args

opts = get_optimal_compile_args(profile="{profile}")

ext_modules = [
    Extension(
        "mymodule",
        ["mymodule.pyx"],
        extra_compile_args=opts['extra_compile_args'],
    )
]"""

    console.print(Panel(
        Syntax(usage_code, "python", theme="monokai", line_numbers=False),
        title="Usage Example",
        border_style="blue"
    ))


def _handle_compile(console: Console, args):
    """Handle compile action."""
    from ..cython_builder import discover_cython_project, build_cython_extensions

    # Get path (defaults to current directory)
    project_path = args.path

    # Dry-run: show build plan
    if args.dry_run:
        try:
            project = discover_cython_project(project_path)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return 1

        # Get optimization flags
        spec = get_system_spec()
        opts = get_optimal_compile_args(spec=spec, profile=args.profile)
        flags = opts.get('extra_compile_args', [])

        # Show build plan
        _show_build_plan(console, project, flags, args)

        console.print("\n[yellow]Run without --dry-run to execute build.[/yellow]")
        return 0

    # Execute build
    console.print(f"[cyan]Building Cython extensions in: {project_path}[/cyan]\n")

    try:
        result = build_cython_extensions(
            path=project_path,
            profile=args.profile,
            jobs=args.jobs,
            dry_run=False,
            clean=args.clean
        )
    except Exception as e:
        console.print(f"[red]Build failed: {e}[/red]")
        return 1

    # Show results
    _show_build_results(console, result)

    return 0 if result['success'] else 1


def _show_build_plan(console: Console, project: dict, flags: list, args):
    """Display build plan for dry-run."""
    config_type = project['config_type']
    pyx_files = project['pyx_files']

    # Build plan panel
    plan_lines = []

    plan_lines.append(f"[cyan]Project:[/cyan] {project['project_root']}")
    plan_lines.append(f"[cyan]Config:[/cyan] {config_type if config_type != 'none' else 'none (requires setup.py or pyproject.toml)'}")

    if pyx_files:
        plan_lines.append("\n[cyan]Files to compile:[/cyan]")
        for i, pyx in enumerate(pyx_files, 1):
            plan_lines.append(f"  {i}. {pyx}")
    else:
        plan_lines.append("\n[red]No .pyx files found![/red]")

    plan_lines.append(f"\n[cyan]Compiler flags ({args.profile} profile):[/cyan]")
    for flag in flags:
        # Get brief explanation
        if flag.startswith('-O'):
            desc = "Optimization level"
        elif flag.startswith('-march=') or flag.startswith('-mcpu='):
            desc = "CPU-specific tuning"
        elif '-math-' in flag:
            desc = "Math function handling"
        elif 'vectorize' in flag:
            desc = "Auto-vectorization"
        elif 'fast-math' in flag:
            desc = "Aggressive FP math"
        else:
            desc = ""

        if desc:
            plan_lines.append(f"  {flag:20} {desc}")
        else:
            plan_lines.append(f"  {flag}")

    if config_type == 'setup.py':
        cmd = f'CFLAGS="{" ".join(flags)}" python setup.py build_ext --inplace'
        if args.jobs and args.jobs > 1:
            cmd += f' -j {args.jobs}'
    elif config_type == 'pyproject.toml':
        cmd = f'CFLAGS="{" ".join(flags)}" pip install -e . --no-build-isolation'
    else:
        cmd = "(No build command - setup.py or pyproject.toml required)"

    plan_lines.append(f"\n[cyan]Build command:[/cyan]\n  {cmd}")

    console.print(Panel(
        "\n".join(plan_lines),
        title="Cython Build Plan",
        border_style="blue"
    ))


def _show_build_results(console: Console, result: dict):
    """Display build results."""
    if result['success']:
        console.print("[green]✓ Build succeeded![/green]\n")

        # Show cleaned files
        if result.get('cleaned'):
            console.print(f"[yellow]Cleaned {len(result['cleaned'])} artifact(s)[/yellow]")

        # Show built files
        files_built = result.get('files_built', [])
        if files_built:
            console.print(f"\n[cyan]Built {len(files_built)} extension(s):[/cyan]")
            for so_file in files_built:
                console.print(f"  • {so_file}")
        else:
            console.print("[yellow]Warning: No .so files found after build[/yellow]")

        # Show build time
        build_time = result.get('build_time_seconds', 0)
        console.print(f"\n[cyan]Build time:[/cyan] {build_time:.1f}s")

    else:
        console.print("[red]✗ Build failed![/red]\n")

        errors = result.get('errors', [])
        for error in errors:
            console.print(f"[red]{error}[/red]")

        if not errors:
            console.print("[red]Unknown error[/red]")
