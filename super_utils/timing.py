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
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .memory import MEMORY_SNAPSHOT, MEMORY_DELTA, _format_bytes, _get_rss_bytes


class _PeakRSSTracker:
    """Background thread that samples RSS to capture peak memory usage."""

    def __init__(self, sample_interval_ms: int = 1):
        self.sample_interval = sample_interval_ms / 1000.0
        self.baseline_rss = 0
        self.peak_rss = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start tracking RSS in background."""
        self.baseline_rss = _get_rss_bytes()
        self.peak_rss = self.baseline_rss
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        """Stop tracking and return peak RSS above baseline."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        # Final sample to catch peak at end
        final_rss = _get_rss_bytes()
        if final_rss > self.peak_rss:
            self.peak_rss = final_rss
        return self.peak_rss - self.baseline_rss

    def _sample_loop(self):
        """Sample RSS until stop event."""
        while not self._stop_event.is_set():
            current_rss = _get_rss_bytes()
            if current_rss > self.peak_rss:
                self.peak_rss = current_rss
            self._stop_event.wait(self.sample_interval)


# Module-level state
_timing_records: List[Dict[str, Any]] = []
_active_timers: Dict[str, Dict[str, Any]] = {}
_active_peak_trackers: Dict[str, _PeakRSSTracker] = {}
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


def TIMING_START(label: str, description: Optional[str] = None, track_peak: bool = False) -> Optional[str]:
    """
    Mark the start of a timed section.

    Parameters
    ----------
    label : str
        Unique label for this timing section (used to match with TIMING_END).
    description : str, optional
        Additional description for this timing point.
    track_peak : bool, default=False
        If True, sample RSS in background thread to capture peak memory usage.
        This reveals memory spikes that delta tracking misses (e.g., NumPy
        allocations that are released before section ends).

    Returns
    -------
    str or None
        The label (for chaining) or None if timing disabled.

    Example
    -------
    >>> TIMING_START("kernel_averaging", "Computing K1 kernels", track_peak=True)
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
        'track_peak': track_peak,
    }

    if track_peak:
        tracker = _PeakRSSTracker(sample_interval_ms=10)
        tracker.start()
        _active_peak_trackers[label] = tracker

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

    # Capture peak RSS if tracking was enabled
    memory_peak = None
    if label in _active_peak_trackers:
        tracker = _active_peak_trackers.pop(label)
        memory_peak = tracker.stop()

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
        'memory_peak': memory_peak,
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
        # Show delta
        msg.append(f"(mem: {'+' if memory_delta >= 0 else ''}{_format_bytes(memory_delta)}",
                   style="red" if memory_delta > 100*1024*1024 else "white")
        # Show peak if tracked
        if memory_peak is not None:
            peak_style = "red bold" if memory_peak > 1024*1024*1024 else "yellow" if memory_peak > 100*1024*1024 else "white"
            msg.append(f", peak: +{_format_bytes(memory_peak)}", style=peak_style)
        msg.append(")")
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
def PROFILE_SECTION(label: str, description: Optional[str] = None, track_peak: bool = False):
    """
    Context manager for combined memory + timing profiling.

    Parameters
    ----------
    label : str
        Unique label for this profiled section.
    description : str, optional
        Additional description for this section.
    track_peak : bool, default=False
        If True, sample RSS in background to capture peak memory usage.
        Essential for detecting memory spikes that are released before
        section ends (e.g., NumPy vectorized operations that allocate
        large intermediates).

    Example
    -------
    >>> with PROFILE_SECTION("kernel_computation", "Computing kernels", track_peak=True):
    ...     result = expensive_function()
    # Output: TIME [12:34:56.789] kernel_computation 4.567s (mem: +7.3 MB, peak: +4.2 GB)
    """
    if not _timing_enabled:
        yield
        return

    TIMING_START(label, description, track_peak=track_peak)
    MEMORY_SNAPSHOT(f"[START] {label}")
    try:
        yield
    finally:
        MEMORY_DELTA(f"[END] {label}")
        TIMING_END(label)
