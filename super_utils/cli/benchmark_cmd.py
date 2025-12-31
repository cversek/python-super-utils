"""
Benchmark CLI commands.

Commands:
- superutils benchmark list       - Discover benchmarks in current directory
- superutils benchmark list-tests - List internal super_utils benchmarks
- superutils benchmark run        - Run benchmarks
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def add_benchmark_parser(subparsers):
    """Add benchmark subcommand parser."""
    parser = subparsers.add_parser(
        'benchmark',
        help='Run and discover benchmarks',
        description='Discover project benchmarks and run performance tests'
    )

    parser.add_argument(
        'action',
        choices=['list', 'list-tests', 'run'],
        nargs='?',
        default='run',
        help='Action to perform (default: run)'
    )

    parser.add_argument(
        '--class', '-c',
        dest='benchmark_class',
        help='Benchmark class(es) to run, comma-separated (default: all)'
    )

    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=5,
        help='Number of benchmark iterations (default: 5)'
    )

    parser.add_argument(
        '--output', '-o',
        help='Save benchmark results to JSON file'
    )

    parser.add_argument(
        '--size', '-s',
        choices=['small', 'medium', 'large'],
        default='medium',
        help='Problem size for benchmarks (default: medium)'
    )

    parser.add_argument(
        '--profile',
        choices=['baseline', 'conservative', 'aggressive'],
        default='conservative',
        help='Optimization profile for system info (default: conservative)'
    )

    return parser


def handle_benchmark(args):
    """Handle benchmark subcommand."""
    console = Console()

    if args.action == 'list':
        return _handle_list_project(console, args)
    elif args.action == 'list-tests':
        return _handle_list_internal(console, args)
    elif args.action == 'run':
        return _handle_run(console, args)
    else:
        console.print(f"[red]Unknown action: {args.action}[/red]")
        return 1


def _handle_list_internal(console: Console, args):
    """List internal super_utils benchmarks."""
    from ..benchmarks import list_benchmarks

    benchmarks = list_benchmarks()

    table = Table(title="Internal Benchmark Tests (super_utils)", show_header=True, header_style="bold cyan")
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
    console.print(f"\n[dim]Total: {len(benchmarks)} internal benchmarks[/dim]")
    console.print("[dim]Run with: superutils benchmark run --class <name>[/dim]")
    return 0


def _handle_list_project(console: Console, args):
    """Discover benchmarks in current directory."""
    from ..benchmarks import discover_project_benchmarks

    cwd = Path.cwd()
    console.print(f"[cyan]Searching for BenchmarkBase subclasses in:[/cyan] {cwd}\n")

    discovered = discover_project_benchmarks(cwd)

    if not discovered:
        console.print("[yellow]No project benchmarks found.[/yellow]")
        console.print("\n[dim]To create a benchmark, define a class that inherits from[/dim]")
        console.print("[dim]super_utils.benchmarks.BenchmarkBase with name, description,[/dim]")
        console.print("[dim]and workload_type class attributes.[/dim]")
        console.print("\n[dim]Use 'superutils benchmark list-tests' for internal benchmarks.[/dim]")
        return 0

    table = Table(title="Discovered Project Benchmarks", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan", width=20)
    table.add_column("Type", style="yellow", width=14)
    table.add_column("Description", style="white", width=30)
    table.add_column("Location", style="dim", width=40)

    for name, info in discovered.items():
        module_path = info.get('module_path', '')
        if len(module_path) > 40:
            module_path = "..." + module_path[-37:]

        table.add_row(
            name,
            info['workload_type'],
            info['description'],
            module_path
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(discovered)} project benchmark(s)[/dim]")
    console.print("[dim]Run with: superutils benchmark run --class <name>[/dim]")
    return 0


def _handle_run(console: Console, args):
    """Run benchmarks."""
    from ..benchmarks import (
        list_benchmarks, run_benchmarks, save_results,
        discover_project_benchmarks, BENCHMARKS
    )
    from ..system_spec import get_system_spec
    from ..cython_optimizer import get_optimal_compile_args

    console.print("[cyan]Running benchmarks...[/cyan]\n")

    # Split comma-separated benchmark classes
    classes = args.benchmark_class.split(',') if args.benchmark_class else None

    # Check if any requested classes are project benchmarks
    project_benchmarks = {}
    if classes:
        discovered = discover_project_benchmarks(Path.cwd())
        for cls_name in classes:
            if cls_name not in BENCHMARKS and cls_name in discovered:
                project_benchmarks[cls_name] = discovered[cls_name]

    try:
        # Determine which internal benchmarks to run
        internal_classes = None
        if classes:
            internal_classes = [c for c in classes if c in BENCHMARKS]
            if not internal_classes:
                internal_classes = None

        # Run internal benchmarks if any
        if internal_classes or (classes is None and not project_benchmarks):
            results = run_benchmarks(
                classes=internal_classes,
                iterations=args.iterations,
                warmup=3,
                size=args.size,
                profile=args.profile,
                include_system_info=True
            )
        else:
            # Initialize empty results for project-only runs
            spec = get_system_spec()
            opts = get_optimal_compile_args(spec=spec, profile=args.profile)

            results = {
                'timestamp': datetime.datetime.now().isoformat(),
                'system': {
                    'cpu': spec.get('hardware', {}).get('cpu_model', 'unknown'),
                    'cores': spec.get('hardware', {}).get('cpu_cores_logical', '?'),
                    'arch': spec.get('hardware', {}).get('architecture', 'unknown'),
                },
                'profile': args.profile,
                'flags': opts.get('extra_compile_args', []),
                'settings': {
                    'iterations': args.iterations,
                    'warmup': 3,
                    'size': args.size,
                },
                'results': {},
                'errors': {},
            }

        # Run project benchmarks
        for name, info in project_benchmarks.items():
            try:
                benchmark_cls = info['cls']
                benchmark = benchmark_cls(size=args.size)
                result = benchmark.benchmark(iterations=args.iterations, warmup=3)
                result['source'] = 'project'
                result['module_path'] = info.get('module_path', '')
                results['results'][name] = result
            except Exception as e:
                if 'errors' not in results:
                    results['errors'] = {}
                results['errors'][name] = f"Error: {e}"

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\nUse 'superutils benchmark list' to discover project benchmarks")
        console.print("Use 'superutils benchmark list-tests' for internal benchmarks")
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
                    table.add_section()
                first_pair = False

                base_time = benchmark_results.get(base, {}).get('mean_ms', 0)

                for name in [base, cython]:
                    if name in benchmark_results:
                        result = benchmark_results[name]
                        added.add(name)
                        valid_mark = "[green]✓[/green]" if result.get('valid') else "[red]✗[/red]"
                        mean_ms = result.get('mean_ms', 0)

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

        # Add unpaired benchmarks (including project benchmarks)
        for name, result in benchmark_results.items():
            if name not in added:
                valid_mark = "[green]✓[/green]" if result.get('valid') else "[red]✗[/red]"
                source_marker = " [dim](project)[/dim]" if result.get('source') == 'project' else ""
                table.add_row(
                    name + source_marker,
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
