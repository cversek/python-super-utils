"""
timing.py
=========

Timing and profiling utilities for tracking execution time across code sections.
Follows the same patterns as memory.py: Rich formatting, timeline tracking,
and JSON export for analysis.

Functions:
----------
- TIMING_START: Mark the start of a timed section.
- TIMING_END: Mark the end and print elapsed time.
- TIMING_REPORT: Generate formatted timing summary report.
- TIMING_TIMELINE: Display accumulated timings as timeline.
- TIMING_RESET: Clear all accumulated timing data.
- TIMING_CONFIGURE: Configure global timing settings.
- TIMING_EXPORT_JSON: Export timing data to JSON file.
- PROFILE_SECTION: Context manager for combined memory + timing.
"""
import os
import time
import json
import inspect
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .memory import MEMORY_SNAPSHOT, MEMORY_DELTA, _format_bytes, _get_rss_bytes


# Module-level state
_timing_records: List[Dict[str, Any]] = []
_active_timers: Dict[str, Dict[str, Any]] = {}
_max_records: int = 1000
_timing_enabled: bool = True


def TIMING_CONFIGURE(enabled: bool = True, max_records: int = 1000):
    """Configure global timing settings."""
    global _timing_enabled, _max_records
    _timing_enabled = enabled
    _max_records = max_records


def TIMING_RESET():
    """Clear all accumulated timing records and active timers."""
    global _timing_records, _active_timers
    _timing_records = []
    _active_timers = {}


def TIMING_START(label: str, description: Optional[str] = None) -> Optional[str]:
    """
    Mark the start of a timed section.

    Parameters
    ----------
    label : str
        Unique label for this timing section (used to match with TIMING_END).
    description : str, optional
        Additional description for this timing point.

    Returns
    -------
    str or None
        The label (for chaining) or None if timing disabled.

    Example
    -------
    >>> TIMING_START("kernel_averaging", "Computing K1 kernels")
    """
    if not _timing_enabled:
        return None

    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)
    location = f"{os.path.basename(info.filename)}:{info.lineno}"

    _active_timers[label] = {
        'start_time': time.perf_counter(),
        'start_datetime': datetime.now().strftime('%H:%M:%S.%f')[:-3],
        'start_location': location,
        'function': info.function,
        'description': description or '',
        'start_rss': _get_rss_bytes(),
    }
    return label


def TIMING_END(label: str, print_result: bool = True) -> Optional[Dict[str, Any]]:
    """
    Mark the end of a timed section and record elapsed time.

    Parameters
    ----------
    label : str
        Label matching a previous TIMING_START call.
    print_result : bool, default=True
        Print timing result to console.

    Returns
    -------
    dict or None
        Timing record with elapsed time, or None if label not found/disabled.

    Example
    -------
    >>> TIMING_END("kernel_averaging")
    """
    global _timing_records

    if not _timing_enabled:
        return None

    if label not in _active_timers:
        Console().print(f"[red]TIMING_END: No active timer '{label}'[/red]")
        return None

    end_time = time.perf_counter()
    start_data = _active_timers.pop(label)

    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)
    end_location = f"{os.path.basename(info.filename)}:{info.lineno}"

    elapsed = end_time - start_data['start_time']
    end_rss = _get_rss_bytes()
    memory_delta = end_rss - start_data['start_rss']

    record = {
        'label': label,
        'description': start_data['description'],
        'start_datetime': start_data['start_datetime'],
        'end_datetime': datetime.now().strftime('%H:%M:%S.%f')[:-3],
        'start_location': start_data['start_location'],
        'end_location': end_location,
        'function': start_data['function'],
        'elapsed_seconds': elapsed,
        'memory_delta': memory_delta,
        'timestamp': time.time(),
    }

    _timing_records.append(record)
    if len(_timing_records) > _max_records:
        _timing_records = _timing_records[-_max_records:]

    if print_result:
        msg = Text()
        msg.append("TIME ", style="bold magenta")
        msg.append(f"[{record['end_datetime']}] ", style="dim")
        msg.append(f"{label} ", style="cyan bold")
        msg.append(f"{elapsed:.3f}s ", style="green" if elapsed < 1.0 else "yellow" if elapsed < 10.0 else "red")
        msg.append(f"(mem: {'+' if memory_delta >= 0 else ''}{_format_bytes(memory_delta)})",
                   style="red" if memory_delta > 100*1024*1024 else "white")
        if start_data['description']:
            msg.append(f" - {start_data['description']}", style="italic dim")
        Console().print(msg)

    return record


def TIMING_REPORT():
    """Generate formatted timing summary report."""
    if not _timing_records:
        Console().print("[dim]No timing records.[/dim]")
        return

    by_label: Dict[str, List[float]] = {}
    for rec in _timing_records:
        by_label.setdefault(rec['label'], []).append(rec['elapsed_seconds'])

    table = Table(title="Timing Report", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=30)
    table.add_column("Count", justify="right", width=8)
    table.add_column("Total", justify="right", width=12)
    table.add_column("Mean", justify="right", width=12)

    total_time = 0.0
    for label, times in sorted(by_label.items()):
        total = sum(times)
        total_time += total
        table.add_row(label, str(len(times)), f"{total:.3f}s", f"{total/len(times):.3f}s")

    Console().print(table)
    Console().print(f"\n[bold]Total:[/bold] {total_time:.3f}s")


def TIMING_TIMELINE(max_entries: int = 20):
    """Display accumulated timing records as timeline."""
    if not _timing_records:
        Console().print("[dim]No timing records.[/dim]")
        return

    entries = _timing_records[-max_entries:]
    table = Table(title=f"Timing Timeline (last {len(entries)})", header_style="bold magenta")
    table.add_column("Time", style="dim", width=12)
    table.add_column("Label", style="cyan", width=25)
    table.add_column("Elapsed", justify="right", width=12)
    table.add_column("Description", style="italic", max_width=25)

    for rec in entries:
        color = "green" if rec['elapsed_seconds'] < 1.0 else "yellow" if rec['elapsed_seconds'] < 10.0 else "red"
        table.add_row(rec['end_datetime'], rec['label'], f"[{color}]{rec['elapsed_seconds']:.3f}s[/{color}]",
                      rec['description'][:25] if rec['description'] else "")
    Console().print(table)


def TIMING_EXPORT_JSON(output_path: str, include_system_spec: bool = True) -> str:
    """
    Export timing data to JSON file.

    Parameters:
        output_path: Path to output JSON file.
        include_system_spec: If True, include system specification for reproducibility.

    Returns:
        The output path.
    """
    by_label: Dict[str, Dict[str, Any]] = {}
    for rec in _timing_records:
        label = rec['label']
        if label not in by_label:
            by_label[label] = {'count': 0, 'total': 0.0, 'times': [], 'mem_deltas': []}
        by_label[label]['count'] += 1
        by_label[label]['total'] += rec['elapsed_seconds']
        by_label[label]['times'].append(rec['elapsed_seconds'])
        by_label[label]['mem_deltas'].append(rec['memory_delta'])

    summary = {}
    for label, data in by_label.items():
        summary[label] = {
            'count': data['count'],
            'total_seconds': data['total'],
            'mean_seconds': data['total'] / data['count'],
            'min_seconds': min(data['times']),
            'max_seconds': max(data['times']),
        }

    export = {'timestamp': datetime.now().isoformat(), 'summary': summary, 'raw': _timing_records}

    # Include system specification for reproducible benchmarks
    if include_system_spec:
        from .system_spec import get_system_spec
        export['system_spec'] = get_system_spec()

    with open(output_path, 'w') as f:
        json.dump(export, f, indent=2)

    Console().print(f"[green]Exported to: {output_path}[/green]")
    return output_path


@contextmanager
def PROFILE_SECTION(label: str, description: Optional[str] = None):
    """
    Context manager for combined memory + timing profiling.

    Example
    -------
    >>> with PROFILE_SECTION("kernel_computation", "Computing kernels"):
    ...     result = expensive_function()
    """
    if not _timing_enabled:
        yield
        return

    TIMING_START(label, description)
    MEMORY_SNAPSHOT(f"[START] {label}")
    try:
        yield
    finally:
        MEMORY_DELTA(f"[END] {label}")
        TIMING_END(label)
