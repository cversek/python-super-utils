"""
super_utils CLI main dispatcher.

Provides subcommands:
- superutils cython [detect|recommend]
- superutils sysspec [show|export]
"""

import sys
import argparse
from typing import List, Optional


def main(argv: Optional[List[str]] = None):
    """
    Main entry point for superutils CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])
    """
    parser = argparse.ArgumentParser(
        prog='superutils',
        description='Utilities for system inspection and Cython optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Cython subcommand
    from .cython_cmd import add_cython_parser
    add_cython_parser(subparsers)

    # Sysspec subcommand
    from .sysspec_cmd import add_sysspec_parser
    add_sysspec_parser(subparsers)

    # Parse args
    args = parser.parse_args(argv)

    # Dispatch
    if args.command == 'cython':
        from .cython_cmd import handle_cython
        return handle_cython(args)
    elif args.command == 'sysspec':
        from .sysspec_cmd import handle_sysspec
        return handle_sysspec(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
