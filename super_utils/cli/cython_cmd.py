"""
Cython optimization CLI commands.

Commands:
- superutils cython detect     - Show hardware + recommended flags
- superutils cython recommend  - Just show recommended flags (no hardware details)
- superutils cython benchmark  - Run optimization benchmarks
- superutils cython compile    - Build Cython extensions with optimized flags
"""

import argparse
import json
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
        choices=['detect', 'recommend', 'benchmark', 'compile'],
        help='Action to perform'
    )

    # Common arguments
    parser.add_argument(
        '--profile',
        choices=['conservative', 'aggressive'],
        default='conservative',
        help='Optimization profile (default: conservative)'
    )

    # Benchmark-specific arguments
    parser.add_argument(
        '--class',
        dest='benchmark_class',
        help='Benchmark class to run (default: all)'
    )

    parser.add_argument(
        '--list',
        dest='list_benchmarks',
        action='store_true',
        help='List available benchmark classes'
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of benchmark iterations (default: 5)'
    )

    parser.add_argument(
        '--output',
        help='Save benchmark results to JSON file'
    )

    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Rebuild Cython extensions before running benchmarks'
    )

    parser.add_argument(
        '--size',
        choices=['small', 'medium', 'large'],
        default='medium',
        help='Problem size for benchmarks (default: medium)'
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
    elif args.action == 'benchmark':
        return _handle_benchmark(console, args)
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


def _handle_benchmark(console: Console, args):
    """Handle benchmark action."""
    from ..benchmarks import list_benchmarks, run_benchmarks, save_results

    # List benchmarks if requested
    if args.list_benchmarks:
        benchmarks = list_benchmarks()

        table = Table(title="Available Benchmark Classes", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="cyan", width=20)
        table.add_column("Type", style="yellow", width=20)
        table.add_column("Description", style="white")

        for name, info in benchmarks.items():
            table.add_row(
                name,
                info['workload_type'],
                info['description']
            )

        console.print(table)
        return 0

    # Rebuild Cython extensions if requested
    if args.rebuild:
        rebuild_result = _rebuild_cython_extensions(console, args.profile)
        if rebuild_result != 0:
            return rebuild_result

    # Run benchmarks
    console.print("[cyan]Running benchmarks...[/cyan]\n")

    classes = [args.benchmark_class] if args.benchmark_class else None

    try:
        results = run_benchmarks(
            classes=classes,
            iterations=args.iterations,
            warmup=3,
            size=args.size,
            profile=args.profile,
            include_system_info=True
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\nUse --list to see available benchmark classes")
        return 1
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        return 1

    # Display results
    _show_benchmark_results(console, results)

    # Save to file if requested
    if args.output:
        try:
            save_results(results, args.output)
            console.print(f"\n[green]Results saved to: {args.output}[/green]")
        except Exception as e:
            console.print(f"\n[red]Failed to save results: {e}[/red]")
            return 1

    return 0


def _rebuild_cython_extensions(console: Console, profile: str) -> int:
    """
    Rebuild Cython extensions with the specified optimization profile.

    Args:
        console: Rich console for output
        profile: Optimization profile ('conservative' or 'aggressive')

    Returns:
        0 on success, non-zero on failure
    """
    import subprocess
    import os
    from pathlib import Path

    console.print(f"[cyan]Rebuilding Cython extensions with {profile} profile...[/cyan]\n")

    # Get optimization flags
    spec = get_system_spec()
    opts = get_optimal_compile_args(spec=spec, profile=profile)
    flags = opts.get('extra_compile_args', [])

    # Find the super-utils package directory
    # Walk up from this file to find setup.py
    current_file = Path(__file__).resolve()
    package_root = current_file.parent.parent.parent  # cli -> super_utils -> super-utils

    setup_py = package_root / 'setup.py'
    if not setup_py.exists():
        console.print(f"[red]Error: Could not find setup.py at {setup_py}[/red]")
        return 1

    console.print(f"[dim]Package root: {package_root}[/dim]")
    console.print(f"[dim]CFLAGS: {' '.join(flags)}[/dim]\n")

    # Set CFLAGS environment variable
    env = os.environ.copy()
    env['CFLAGS'] = ' '.join(flags)

    # Run build_ext --inplace
    try:
        result = subprocess.run(
            ['python', 'setup.py', 'build_ext', '--inplace'],
            cwd=str(package_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            console.print("[red]Build failed![/red]")
            if result.stderr:
                console.print(f"[red]{result.stderr}[/red]")
            return 1

        console.print("[green]Build succeeded![/green]\n")

        # Show which .so files were built
        so_files = list(package_root.glob('**/*.so'))
        if so_files:
            console.print("[cyan]Built extensions:[/cyan]")
            for so_file in so_files:
                rel_path = so_file.relative_to(package_root)
                console.print(f"  {rel_path}")
            console.print()

        return 0

    except subprocess.TimeoutExpired:
        console.print("[red]Build timed out after 5 minutes[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Build error: {e}[/red]")
        return 1


def _show_benchmark_results(console: Console, results: dict):
    """Display benchmark results."""
    # System info
    if results.get('system'):
        sys_info = results['system']
        console.print(Panel(
            f"[cyan]CPU:[/cyan] {sys_info['cpu']}\n"
            f"[cyan]Cores:[/cyan] {sys_info['cores']}\n"
            f"[cyan]Architecture:[/cyan] {sys_info['arch']}\n"
            f"[cyan]Profile:[/cyan] {results['profile']}",
            title="System Information",
            border_style="blue"
        ))
        console.print()

    # Results table
    benchmark_results = results.get('results', {})

    if benchmark_results:
        table = Table(title="Benchmark Results", show_header=True, header_style="bold cyan")
        table.add_column("Benchmark", style="cyan", width=20, no_wrap=True)
        table.add_column("Type", style="yellow", width=14)
        table.add_column("Mean (ms)", justify="right", style="green", width=10)
        table.add_column("Std (ms)", justify="right", style="white", width=8)
        table.add_column("Min (ms)", justify="right", style="white", width=10)
        table.add_column("Max (ms)", justify="right", style="white", width=10)
        table.add_column("Speedup", justify="right", style="magenta", width=8)
        table.add_column("Valid", justify="center")

        # Define benchmark pairs for grouping
        pairs = [
            ('streaming', 'cython_streaming'),
            ('wavelet', 'cython_wavelet'),
            ('branch', 'cython_branch'),
            ('linalg', 'cython_linalg'),
            ('interp', 'cython_interp'),
            ('mfvep', 'cython_mfvep'),
        ]

        # Build ordered list with separators
        added = set()
        first_pair = True
        for base, cython in pairs:
            if base in benchmark_results or cython in benchmark_results:
                if not first_pair:
                    table.add_section()  # Thin horizontal line
                first_pair = False

                base_time = benchmark_results.get(base, {}).get('mean_ms', 0)

                for name in [base, cython]:
                    if name in benchmark_results:
                        result = benchmark_results[name]
                        added.add(name)
                        valid_mark = "[green]✓[/green]" if result.get('valid') else "[red]✗[/red]"
                        mean_ms = result.get('mean_ms', 0)

                        # Calculate speedup (base/cython)
                        if name.startswith('cython_') and base_time > 0 and mean_ms > 0:
                            speedup = f"{base_time / mean_ms:.1f}x"
                        else:
                            speedup = "-"

                        table.add_row(
                            name,
                            result.get('workload_type', '?'),
                            f"{mean_ms:.2f}",
                            f"{result.get('std_ms', 0):.2f}",
                            f"{result.get('min_ms', 0):.2f}",
                            f"{result.get('max_ms', 0):.2f}",
                            speedup,
                            valid_mark
                        )

        # Add any unpaired benchmarks at the end
        for name, result in benchmark_results.items():
            if name not in added:
                valid_mark = "[green]✓[/green]" if result.get('valid') else "[red]✗[/red]"
                table.add_row(
                    name,
                    result.get('workload_type', '?'),
                    f"{result.get('mean_ms', 0):.2f}",
                    f"{result.get('std_ms', 0):.2f}",
                    f"{result.get('min_ms', 0):.2f}",
                    f"{result.get('max_ms', 0):.2f}",
                    "-",
                    valid_mark
                )

        console.print(table)

    # Errors
    if results.get('errors'):
        console.print()
        for name, error in results['errors'].items():
            console.print(f"[yellow]Warning:[/yellow] {name}: {error}")

    # Recommendation
    if results.get('recommendation'):
        console.print()
        console.print(Panel(
            results['recommendation'],
            title="Optimization Recommendation",
            border_style="green"
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
