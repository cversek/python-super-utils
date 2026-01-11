"""
System specification CLI commands.

Commands:
- superutils sysspec show           - Display system specification
- superutils sysspec export <file>  - Export to JSON file
"""

import argparse
from rich.console import Console

from ..system_spec import get_system_spec, print_system_report, export_system_spec


def add_sysspec_parser(subparsers):
    """Add sysspec subcommand parser."""
    parser = subparsers.add_parser(
        'sysspec',
        help='System specification display/export',
        description='Display or export detailed system hardware and software configuration'
    )

    parser.add_argument(
        'action',
        choices=['show', 'export'],
        help='show: display to console | export: save to JSON file'
    )

    parser.add_argument(
        'file',
        nargs='?',
        help='Output file path (required for export)'
    )

    return parser


def handle_sysspec(args):
    """Handle sysspec subcommand."""
    console = Console()

    if args.action == 'show':
        # Display to console
        spec = get_system_spec()
        print_system_report(spec)
        return 0

    elif args.action == 'export':
        # Export to JSON
        if not args.file:
            console.print("[red]Error:[/red] export action requires a file path", style="bold")
            console.print("Usage: superutils sysspec export <file>")
            return 1

        try:
            spec = get_system_spec()
            export_system_spec(args.file, spec)
            console.print(f"[green]✓[/green] System spec exported to: {args.file}")
            return 0
        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to export: {e}", style="bold")
            return 1

    return 0
