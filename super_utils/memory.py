"""
memory.py
=========

Memory profiling utilities for tracking memory usage across code execution.
Goes beyond basic tracemalloc by providing timeline tracking, delta analysis,
trend detection, and Rich-formatted reporting.

Functions:
----------
- MEMORY_SNAPSHOT: Capture memory state at a code point with frame context.
- MEMORY_REPORT: Generate formatted memory usage report.
- MEMORY_TIMELINE: Display accumulated snapshots as timeline.
- MEMORY_RESET: Clear all accumulated snapshots.
- MEMORY_CONFIGURE: Configure global memory tracking settings.
- MEMORY_DELTA: Show memory change since last snapshot.
"""
# standard imports
import sys
import os
import gc
import time
import tracemalloc
import inspect
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field

# third party imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


# Module-level state
_memory_snapshots: List[Dict[str, Any]] = []
_max_snapshots: int = 1000
_memory_enabled: bool = True
_tracemalloc_started: bool = False
_peak_rss: int = 0
_peak_traced: int = 0


def _get_rss_bytes() -> int:
    """Get process RSS (Resident Set Size) from /proc/self/status."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024  # KB to bytes
    except (FileNotFoundError, PermissionError):
        pass
    return 0


def _ensure_tracemalloc():
    """Start tracemalloc if not already running."""
    global _tracemalloc_started
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        _tracemalloc_started = True


def _format_bytes(size: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def MEMORY_CONFIGURE(enabled: bool = True, max_snapshots: int = 1000):
    """
    Configure global memory tracking settings.

    Parameters:
    -----------
    enabled : bool, default=True
        Enable or disable memory tracking globally.
    max_snapshots : int, default=1000
        Maximum number of snapshots to retain in timeline.

    Example:
    --------
    >>> MEMORY_CONFIGURE(enabled=True, max_snapshots=500)
    """
    global _memory_enabled, _max_snapshots
    _memory_enabled = enabled
    _max_snapshots = max_snapshots


def MEMORY_RESET():
    """
    Clear all accumulated memory snapshots and reset peak tracking.

    Example:
    --------
    >>> MEMORY_RESET()
    """
    global _memory_snapshots, _peak_rss, _peak_traced
    _memory_snapshots = []
    _peak_rss = 0
    _peak_traced = 0
    gc.collect()


def MEMORY_SNAPSHOT(description: Optional[str] = None, force_gc: bool = False):
    """
    Capture current memory usage with caller's frame context.

    Parameters:
    -----------
    description : str, optional
        Description of what triggered this snapshot.
    force_gc : bool, default=False
        Run garbage collection before measuring.

    Returns:
    --------
    dict
        Snapshot data with RSS, traced memory, and frame info.

    Example:
    --------
    >>> MEMORY_SNAPSHOT("Before heavy computation")
    """
    global _memory_snapshots, _peak_rss, _peak_traced

    if not _memory_enabled:
        return None

    _ensure_tracemalloc()

    if force_gc:
        gc.collect()

    # Get caller's frame info
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)
    location = f"{os.path.basename(info.filename)}:{info.lineno}"
    function = info.function

    # Get memory metrics
    rss = _get_rss_bytes()
    traced_current, traced_peak = tracemalloc.get_traced_memory()

    # Update peaks
    if rss > _peak_rss:
        _peak_rss = rss
    if traced_peak > _peak_traced:
        _peak_traced = traced_peak

    # Calculate delta from previous snapshot
    delta_rss = 0
    delta_traced = 0
    if _memory_snapshots:
        prev = _memory_snapshots[-1]
        delta_rss = rss - prev['rss']
        delta_traced = traced_current - prev['traced_current']

    snapshot = {
        'timestamp': time.time(),
        'datetime': datetime.now().strftime('%H:%M:%S.%f')[:-3],
        'location': location,
        'function': function,
        'description': description or '',
        'rss': rss,
        'traced_current': traced_current,
        'traced_peak': traced_peak,
        'delta_rss': delta_rss,
        'delta_traced': delta_traced,
    }

    _memory_snapshots.append(snapshot)

    # Trim if over max
    if len(_memory_snapshots) > _max_snapshots:
        _memory_snapshots = _memory_snapshots[-_max_snapshots:]

    return snapshot


def MEMORY_DELTA(description: Optional[str] = None):
    """
    Take snapshot and print delta since last snapshot.

    Parameters:
    -----------
    description : str, optional
        Description of this checkpoint.

    Example:
    --------
    >>> MEMORY_DELTA("After loading dataset")
    """
    snapshot = MEMORY_SNAPSHOT(description)
    if snapshot is None:
        return

    console = Console()

    delta_sign = "+" if snapshot['delta_rss'] >= 0 else ""
    delta_color = "red" if snapshot['delta_rss'] > 0 else "green"

    msg = Text()
    msg.append("MEM ", style="bold cyan")
    msg.append(f"[{snapshot['datetime']}] ", style="dim")
    msg.append(f"{snapshot['location']} ", style="yellow")
    msg.append(f"RSS={_format_bytes(snapshot['rss'])} ", style="white")
    msg.append(f"({delta_sign}{_format_bytes(snapshot['delta_rss'])})", style=delta_color)
    if description:
        msg.append(f" - {description}", style="italic")

    console.print(msg)


def MEMORY_REPORT(verbose: bool = False, output_file: Optional[str] = None):
    """
    Generate formatted memory usage report.

    Parameters:
    -----------
    verbose : bool, default=False
        Include additional internal details.
    output_file : str, optional
        Write report to file in addition to console.

    Example:
    --------
    >>> MEMORY_REPORT()
    >>> MEMORY_REPORT(verbose=True, output_file="memory.log")
    """
    global _peak_rss, _peak_traced

    # Take a snapshot first
    snapshot = MEMORY_SNAPSHOT("MEMORY_REPORT")
    if snapshot is None:
        return

    console = Console()

    # Build report table
    table = Table(title="Memory Report", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="magenta", justify="right")

    table.add_row("Current RSS", _format_bytes(snapshot['rss']))
    table.add_row("Current Traced", _format_bytes(snapshot['traced_current']))
    table.add_row("Peak RSS (session)", _format_bytes(_peak_rss))
    table.add_row("Peak Traced (session)", _format_bytes(_peak_traced))
    table.add_row("Snapshots Recorded", str(len(_memory_snapshots)))

    if _memory_snapshots and len(_memory_snapshots) > 1:
        first = _memory_snapshots[0]
        total_delta = snapshot['rss'] - first['rss']
        table.add_row("Total RSS Change", _format_bytes(total_delta))

    if verbose and _memory_snapshots:
        table.add_row("", "")
        table.add_row("First Snapshot", _memory_snapshots[0]['datetime'])
        table.add_row("Last Snapshot", snapshot['datetime'])

    console.print()
    console.print(table)
    console.print()

    # Write to file if requested
    if output_file:
        with open(output_file, 'a') as f:
            f.write(f"\n--- Memory Report at {snapshot['datetime']} ---\n")
            f.write(f"RSS: {_format_bytes(snapshot['rss'])}\n")
            f.write(f"Traced: {_format_bytes(snapshot['traced_current'])}\n")
            f.write(f"Peak RSS: {_format_bytes(_peak_rss)}\n")
            f.write(f"Peak Traced: {_format_bytes(_peak_traced)}\n")


def MEMORY_TIMELINE(max_entries: int = 20, show_deltas: bool = True):
    """
    Display accumulated memory snapshots as formatted timeline.

    Parameters:
    -----------
    max_entries : int, default=20
        Limit timeline to N most recent snapshots.
    show_deltas : bool, default=True
        Show memory change between snapshots.

    Example:
    --------
    >>> MEMORY_TIMELINE()
    >>> MEMORY_TIMELINE(max_entries=50, show_deltas=True)
    """
    if not _memory_snapshots:
        console = Console()
        console.print("[dim]No memory snapshots recorded. Use MEMORY_SNAPSHOT() first.[/dim]")
        return

    console = Console()

    # Get last N entries
    entries = _memory_snapshots[-max_entries:]

    table = Table(title=f"Memory Timeline (last {len(entries)} snapshots)",
                  show_header=True, header_style="bold cyan")
    table.add_column("Time", style="dim", width=12)
    table.add_column("Location", style="yellow", width=30)
    table.add_column("RSS", justify="right", width=12)
    if show_deltas:
        table.add_column("Delta", justify="right", width=12)
    table.add_column("Description", style="italic", max_width=30)

    for snap in entries:
        delta_str = ""
        if show_deltas:
            sign = "+" if snap['delta_rss'] >= 0 else ""
            color = "red" if snap['delta_rss'] > 10*1024*1024 else "green" if snap['delta_rss'] < 0 else "white"
            delta_str = f"[{color}]{sign}{_format_bytes(snap['delta_rss'])}[/{color}]"

        row = [
            snap['datetime'],
            snap['location'],
            _format_bytes(snap['rss']),
        ]
        if show_deltas:
            row.append(delta_str)
        row.append(snap['description'][:30] if snap['description'] else "")

        table.add_row(*row)

    console.print()
    console.print(table)

    # Show trend analysis
    if len(_memory_snapshots) >= 3:
        first_rss = _memory_snapshots[0]['rss']
        last_rss = _memory_snapshots[-1]['rss']
        trend = last_rss - first_rss
        trend_str = f"+{_format_bytes(trend)}" if trend >= 0 else _format_bytes(trend)
        trend_color = "red" if trend > 50*1024*1024 else "yellow" if trend > 0 else "green"
        console.print(f"\n[bold]Trend:[/bold] [{trend_color}]{trend_str}[/{trend_color}] over {len(_memory_snapshots)} snapshots")
    console.print()
