"""
memory.py
=========

Memory profiling utilities for tracking memory usage across code execution.
Goes beyond basic tracemalloc by providing timeline tracking, delta analysis,
trend detection, and Rich-formatted reporting.

Point-in-Time Functions:
------------------------
- MEMORY_SNAPSHOT: Capture memory state at a code point with frame context.
- MEMORY_REPORT: Generate formatted memory usage report.
- MEMORY_TIMELINE: Display accumulated snapshots as timeline.
- MEMORY_RESET: Clear all accumulated snapshots.
- MEMORY_CONFIGURE: Configure global memory tracking settings.
- MEMORY_DELTA: Show memory change since last snapshot.

Threaded Peak Tracking (for transient allocations):
---------------------------------------------------
- MEMORY_PEAK_START: Start background thread sampling RSS at intervals.
- MEMORY_PEAK_STOP: Stop tracking, return peak memory above baseline.
- MEMORY_PEAK_CONTEXT: Context manager wrapping start/stop.
- PeakMemoryTracker: Class for manual control of peak tracking.

Use threaded tracking when memory may be freed before measurement (e.g.,
NumPy vectorized ops with large temporaries). Use snapshots when memory
persists until measurement (e.g., streaming/accumulator patterns).
"""
# standard imports
import sys
import os
import gc
import time
import tracemalloc
import inspect
import threading
import warnings
from contextlib import contextmanager

# third-party imports for memory tracking
import psutil
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field

# Optional numpy import for sample collection
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

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

# Module-level state for peak memory tracking
_active_peak_trackers: Dict[str, "PeakMemoryTracker"] = {}


def _get_rss_bytes() -> int:
    """Get process RSS (Resident Set Size) - cross-platform via psutil."""
    return psutil.Process().memory_info().rss


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


# =============================================================================
# Peak Memory Tracking API
# =============================================================================

class PeakMemoryTracker:
    """Background thread that samples RSS to capture peak memory."""

    def __init__(self, sample_interval_ms: int = 10, collect_samples: bool = False):
        """
        Args:
            sample_interval_ms: Sampling interval in milliseconds (default: 10)
            collect_samples: If True, store all samples for later analysis
        """
        self._sample_interval_ms = sample_interval_ms
        self._collect_samples = collect_samples
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._baseline_bytes: int = 0
        self._peak_bytes: int = 0
        self._sample_count: int = 0
        self._samples_list: List[Tuple[float, int]] = []  # (timestamp, rss_bytes)

    def start(self) -> None:
        """Begin background sampling. Records baseline RSS."""
        self._baseline_bytes = _get_rss_bytes()
        self._peak_bytes = self._baseline_bytes
        self._sample_count = 0
        self._samples_list = []
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        """Internal sampling loop run by background thread."""
        interval_sec = self._sample_interval_ms / 1000.0

        while not self._stop_event.is_set():
            current_rss = _get_rss_bytes()
            self._sample_count += 1

            if current_rss > self._peak_bytes:
                self._peak_bytes = current_rss

            if self._collect_samples:
                self._samples_list.append((time.time(), current_rss))

            self._stop_event.wait(interval_sec)

    def stop(self) -> int:
        """Stop sampling. Returns peak delta in bytes."""
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Take final sample after stop signal to catch last-moment peaks
        final_rss = _get_rss_bytes()
        self._sample_count += 1
        if final_rss > self._peak_bytes:
            self._peak_bytes = final_rss
        if self._collect_samples:
            self._samples_list.append((time.time(), final_rss))

        return self._peak_bytes - self._baseline_bytes

    @property
    def peak_mb(self) -> float:
        """Peak memory above baseline in MB."""
        return (self._peak_bytes - self._baseline_bytes) / (1024 * 1024)

    @property
    def sample_count(self) -> int:
        """Number of samples taken."""
        return self._sample_count

    @property
    def baseline_bytes(self) -> int:
        """RSS at start."""
        return self._baseline_bytes

    @property
    def peak_bytes(self) -> int:
        """Maximum RSS observed."""
        return self._peak_bytes

    @property
    def samples(self) -> Optional[Any]:
        """If collect_samples=True, structured array with (timestamp, rss_bytes).
        Returns None if collect_samples=False or numpy unavailable.
        Warns if numpy unavailable and samples were requested."""
        if not self._collect_samples:
            return None

        if not _HAS_NUMPY:
            warnings.warn(
                "numpy not available; cannot return samples as structured array. "
                "Access raw samples via tracker._samples_list if needed.",
                RuntimeWarning
            )
            return None

        if not self._samples_list:
            return np.array([], dtype=[('timestamp', 'f8'), ('rss_bytes', 'i8')])

        return np.array(
            self._samples_list,
            dtype=[('timestamp', 'f8'), ('rss_bytes', 'i8')]
        )


def MEMORY_PEAK_START(label: str, sample_interval_ms: int = 10, collect_samples: bool = False) -> None:
    """
    Start background thread to track peak RSS memory.

    Args:
        label: Unique identifier for this tracking session
        sample_interval_ms: Sampling interval in milliseconds (default: 10)
        collect_samples: If True, store all samples for later analysis

    Raises:
        ValueError: If label already active
    """
    if label in _active_peak_trackers:
        raise ValueError(f"Label '{label}' is already active. Call MEMORY_PEAK_STOP('{label}') first.")

    tracker = PeakMemoryTracker(
        sample_interval_ms=sample_interval_ms,
        collect_samples=collect_samples
    )
    tracker.start()
    _active_peak_trackers[label] = tracker


def MEMORY_PEAK_STOP(label: str) -> Dict[str, Any]:
    """
    Stop tracking and return results.

    Args:
        label: The label passed to MEMORY_PEAK_START

    Returns:
        {
            "peak_mb": float,        # Peak above baseline in MB
            "sample_count": int,     # Number of samples taken
            "baseline_mb": float,    # Baseline RSS in MB
            "peak_bytes": int        # Absolute peak RSS in bytes
        }

    Raises:
        KeyError: If label not found in active trackers
    """
    if label not in _active_peak_trackers:
        raise KeyError(f"Label '{label}' not found in active trackers.")

    tracker = _active_peak_trackers.pop(label)
    tracker.stop()

    return {
        "peak_mb": tracker.peak_mb,
        "sample_count": tracker.sample_count,
        "baseline_mb": tracker.baseline_bytes / (1024 * 1024),
        "peak_bytes": tracker.peak_bytes,
    }


@contextmanager
def MEMORY_PEAK_CONTEXT(label: str, sample_interval_ms: int = 10, collect_samples: bool = False):
    """
    Context manager for peak memory tracking.

    Args:
        label: Unique identifier for this tracking session
        sample_interval_ms: Sampling interval in milliseconds
        collect_samples: If True, store all samples

    Yields:
        PeakMemoryTracker: The tracker object (can access peak_mb, sample_count, etc.)

    Example:
        with MEMORY_PEAK_CONTEXT("kernel_run") as tracker:
            result = expensive_function()
        print(f"Peak: {tracker.peak_mb:.1f} MB over {tracker.sample_count} samples")
    """
    tracker = PeakMemoryTracker(
        sample_interval_ms=sample_interval_ms,
        collect_samples=collect_samples
    )
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()
