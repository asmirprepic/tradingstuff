import argparse
import contextlib
import os
import queue
import re
import shlex
import sys
import threading
import time
from collections import deque
from importlib import import_module

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


COMMANDS = {
    "quality": ("scripts.run_data_quality", "Audit OHLCV data and create an approved ticker file."),
    "technical": ("scripts.run_technical_agents", "Run technical agents and build shortlist CSVs."),
    "ml": ("scripts.run_ml_agents", "Train or load selected ML agents."),
    "dashboard": ("scripts.technical_dashboard", "Build the technical HTML dashboard."),
}

ALIASES = {
    "tech": "technical",
    "machine-learning": "ml",
    "data-quality": "quality",
}


def build_parser():
    command_help = "\n".join(
        f"  {name:<10} {description}" for name, (_, description) in COMMANDS.items()
    )
    parser = argparse.ArgumentParser(
        prog="trading-cli",
        description="Small command line entry point for the trading workflow.",
        epilog=(
            "Commands:\n"
            f"{command_help}\n\n"
            "Examples:\n"
            "  python -m scripts.trading_cli quality --use-synthetic --tickers AAPL,MSFT\n"
            "  python -m scripts.trading_cli technical --tickers-file data_quality_approved_tickers.csv --agents momentum,rsi\n"
            "  python -m scripts.trading_cli ml --tickers AAPL,MSFT --agents qda,spline_logistic\n"
            "  python -m scripts.trading_cli dashboard --latest-manifest-dir ."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", help="menu, quality, technical, ml, or dashboard")
    return parser


def dispatch(command, command_args):
    command = ALIASES.get(command, command)
    if command not in COMMANDS:
        available = ", ".join(COMMANDS)
        raise SystemExit(f"Unknown command '{command}'. Available: {available}")
    if command_args[:1] == ["--"]:
        command_args = command_args[1:]
    module = import_module(COMMANDS[command][0])
    return module.main(command_args)


TOOL_CARDS = (
    ("quality", "DATA QUALITY", "Audit OHLCV integrity", "missing values / stale prices / bad bars"),
    ("technical", "TECHNICAL", "Run indicator agents", "shortlist / consensus / family summary"),
    ("ml", "MACHINE LEARNING", "Train or load models", "probabilities / intervals / anomaly flags"),
    ("dashboard", "REPORT DASHBOARD", "Build the HTML report", "latest timestamped technical run"),
)


class BackRequested(Exception):
    pass


def _prompt(input_fn, prompt, default="", allow_back=True):
    suffix = f" [{default}]" if default else ""
    value = input_fn(f"{prompt}{suffix}: ").strip()
    if allow_back and value.lower() in ("b", "back", "esc"):
        raise BackRequested
    return value or default


def build_runner_args(command, universe, synthetic=False, lookback="260", agents="", extra=""):
    if command == "dashboard":
        command_args = ["--latest-manifest-dir", universe or "."]
    else:
        if universe.lower().endswith((".csv", ".txt")):
            command_args = ["--tickers-file", universe]
        else:
            command_args = ["--tickers", universe]
        if synthetic:
            command_args.append("--use-synthetic")
        else:
            command_args.extend(["--lookback-days", str(lookback)])
        if command == "technical":
            command_args.extend(["--agents", agents or "all"])
        elif command == "ml":
            command_args.extend(["--agents", agents or "lightweight"])
    if extra:
        command_args.extend(shlex.split(extra, posix=True))
    return command_args


def render_terminal_dashboard(selected=0, status="READY"):
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    cards = []
    for index, (_, title, subtitle, detail) in enumerate(TOOL_CARDS):
        active = index == selected
        body = Group(
            Text(title, style="bold white" if active else "bold #d7e4dd"),
            Text(subtitle, style="#f0a06b" if active else "#9badA4"),
            Text(detail, style="dim"),
        )
        cards.append(
            Panel(
                body,
                height=7,
                border_style="bold #e66f3c" if active else "#345248",
                style="on #19352c" if active else "on #10251f",
                padding=(1, 2),
            )
        )
    grid.add_row(cards[0], cards[1])
    grid.add_row(cards[2], cards[3])

    header = Text()
    header.append("MARKET LAB", style="bold #f3efe4")
    header.append("  /  OPERATIONS CONSOLE", style="bold #e66f3c")
    footer = Table.grid(expand=True)
    footer.add_column(justify="left")
    footer.add_column(justify="right")
    footer.add_row(
        "[bold]ARROWS[/] navigate   [bold]ENTER[/] launch   [bold]Q[/] quit",
        f"STATUS  [bold #e66f3c]{status}[/]",
    )
    return Panel(
        Group(Align.left(header), Text("Select a workflow", style="dim"), grid, footer),
        border_style="#527c6e",
        padding=(1, 2),
        title="[bold]TRADINGSTUFF[/]",
        subtitle="local research tools",
    )


def _matching_indices(items, query):
    normalized = query.strip().casefold()
    if not normalized:
        return list(range(len(items)))
    return [index for index, item in enumerate(items) if normalized in str(item).casefold()]


def render_selector(title, items, selected, cursor, offset, page_size=12, query=""):
    table = Table.grid(expand=True)
    table.add_column(width=4)
    table.add_column(ratio=1)
    matches = _matching_indices(items, query)
    visible_indices = matches[offset:offset + page_size]
    for visible_position, item_index in enumerate(visible_indices, start=offset):
        active = visible_position == cursor
        marker = "[x]" if item_index in selected else "[ ]"
        style = "bold white on #285244" if active else "#c6d6cf"
        table.add_row(Text(marker, style=style), Text(str(items[item_index]), style=style), style=style)
    if not matches:
        table.add_row("", Text("No matches", style="italic dim"))
    for _ in range(page_size - max(1 if not matches else 0, len(visible_indices))):
        table.add_row("", "")

    position = f"{cursor + 1}/{len(matches)}" if matches else "0/0"
    filter_text = f"Filter: {query}" if query else "Filter: all"
    return Panel(
        Group(
            Text(title, style="bold #f0a06b"),
            Text(f"Selected {len(selected)} of {len(items)}  |  {filter_text}", style="dim"),
            table,
            Text("/ search   C clear   ARROWS scroll   SPACE toggle   A/N matches   ENTER accept   B back", style="bold"),
        ),
        title="[bold]SELECTOR[/]",
        subtitle=position,
        border_style="#527c6e",
        padding=(1, 2),
    )


def select_items(
    title,
    items,
    initially_selected=None,
    console=None,
    key_reader=None,
    search_input_fn=None,
    page_size=12,
):
    items = list(dict.fromkeys(items))
    if not items:
        return []
    console = console or Console()
    key_reader = key_reader or _read_key
    search_input_fn = search_input_fn or console.input
    initial_values = set(initially_selected) if initially_selected is not None else None
    selected = set(range(len(items))) if initial_values is None else {
        index for index, item in enumerate(items) if item in initial_values
    }
    cursor = 0
    offset = 0
    query = ""
    with Live(
        render_selector(title, items, selected, cursor, offset, page_size, query),
        console=console,
        screen=True,
        auto_refresh=False,
    ) as live:
        while True:
            matches = _matching_indices(items, query)
            live.update(render_selector(title, items, selected, cursor, offset, page_size, query), refresh=True)
            key = key_reader()
            if key in ("B", "ESC", "Q"):
                return None
            if key == "UP":
                cursor = max(0, cursor - 1)
            elif key == "DOWN":
                cursor = min(max(0, len(matches) - 1), cursor + 1)
            elif key == "PGUP":
                cursor = max(0, cursor - page_size)
            elif key == "PGDN":
                cursor = min(max(0, len(matches) - 1), cursor + page_size)
            elif key == "SPACE" and matches:
                item_index = matches[cursor]
                if item_index in selected:
                    selected.remove(item_index)
                else:
                    selected.add(item_index)
            elif key == "A":
                selected.update(matches)
            elif key == "N":
                selected.difference_update(matches)
            elif key == "/":
                live.stop()
                query = search_input_fn("Search (empty shows all): ").strip()
                cursor = 0
                offset = 0
                live.start(refresh=True)
            elif key == "C":
                query = ""
                cursor = 0
                offset = 0
            elif key == "ENTER":
                return [item for index, item in enumerate(items) if index in selected]
            offset = min(max(0, cursor - page_size + 1), max(0, len(matches) - page_size))


def select_one(title, options, console=None, key_reader=None):
    options = list(options)
    if not options:
        return None
    console = console or Console()
    key_reader = key_reader or _read_key
    cursor = 0
    with Live(
        render_selector(title, options, {cursor}, cursor, 0, page_size=min(12, len(options))),
        console=console,
        screen=True,
        auto_refresh=False,
    ) as live:
        while True:
            live.update(
                render_selector(title, options, {cursor}, cursor, 0, page_size=min(12, len(options))),
                refresh=True,
            )
            key = key_reader()
            if key in ("B", "ESC", "Q"):
                return None
            if key == "UP":
                cursor = max(0, cursor - 1)
            elif key == "DOWN":
                cursor = min(len(options) - 1, cursor + 1)
            elif key == "ENTER":
                return options[cursor]


def ticker_groups(frame):
    ticker_column = next(
        (column for column in ("Stock", "Ticker", "ticker", "Tickers", "tickers") if column in frame.columns),
        frame.columns[0] if len(frame.columns) else None,
    )
    if ticker_column is None:
        return {}
    group_columns = [
        column for column in ("Status", "ShortlistTier", "Sector", "Industry", "MarketCapGroup")
        if column in frame.columns
    ]
    groups = {}
    for group_column in group_columns:
        values = {}
        valid = frame.dropna(subset=[ticker_column, group_column])
        for name, group in valid.groupby(group_column, sort=True):
            values[str(name)] = group[ticker_column].astype(str).drop_duplicates().tolist()
        if values:
            groups[group_column] = values
    return groups


def _groups_for_universe(universe):
    if not universe.lower().endswith(".csv") or not os.path.exists(os.path.expanduser(universe)):
        return {}
    import pandas as pd

    return ticker_groups(pd.read_csv(os.path.expanduser(universe)))


def choose_ticker_universe(universe, console=None, key_reader=None):
    console = console or Console()
    key_reader = key_reader or _read_key
    tickers = _ticker_choices(universe)
    if not tickers:
        return universe
    if len(tickers) <= 30:
        selected = select_items(
            "TICKERS", tickers, initially_selected=tickers, console=console, key_reader=key_reader
        )
        if selected is None:
            raise BackRequested
        return universe if len(selected) == len(tickers) else ",".join(selected)

    groups = _groups_for_universe(universe)
    actions = [f"Use all {len(tickers)} tickers"]
    actions.extend(f"Filter by {column}" for column in groups)
    actions.extend(("Browse individual tickers", "Back"))
    action = select_one("TICKER UNIVERSE", actions, console=console, key_reader=key_reader)
    if action is None or action == "Back":
        raise BackRequested
    if action.startswith("Use all"):
        return universe
    if action == "Browse individual tickers":
        selected = select_items(
            "TICKERS", tickers, initially_selected=tickers, console=console, key_reader=key_reader
        )
        if selected is None:
            raise BackRequested
        return universe if len(selected) == len(tickers) else ",".join(selected)

    group_column = action.removeprefix("Filter by ")
    group_values = groups[group_column]
    default_groups = ["Pass"] if "Pass" in group_values else list(group_values)
    selected_groups = select_items(
        group_column.upper(),
        list(group_values),
        initially_selected=default_groups,
        console=console,
        key_reader=key_reader,
    )
    if selected_groups is None:
        raise BackRequested
    selected_set = set(selected_groups)
    allowed_tickers = {
        ticker for group in selected_set for ticker in group_values[group]
    }
    selected_tickers = [ticker for ticker in tickers if ticker in allowed_tickers]
    if not selected_tickers:
        raise BackRequested
    return ",".join(selected_tickers)


def _read_key():
    if os.name == "nt":
        import msvcrt

        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            return {
                "H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT",
                "I": "PGUP", "Q": "PGDN",
            }.get(msvcrt.getwch(), "")
        if key == "\r":
            return "ENTER"
        if key == " ":
            return "SPACE"
        if key == "\x1b":
            return "ESC"
        return key.upper()

    import select
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        key = sys.stdin.read(1)
        if key == "\x1b" and select.select([sys.stdin], [], [], 0.05)[0]:
            sequence = sys.stdin.read(2)
            return {"[A": "UP", "[B": "DOWN", "[D": "LEFT", "[C": "RIGHT"}.get(sequence, "")
        if key == "\x1b":
            return "ESC"
        if key in ("\r", "\n"):
            return "ENTER"
        if key == " ":
            return "SPACE"
        return key.upper()
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _available_agents(command):
    if command == "technical":
        return list(import_module("scripts.run_technical_agents").AGENT_ORDER), None
    if command == "ml":
        module = import_module("scripts.run_ml_agents")
        return list(module.AGENT_ORDER), list(module.AGENT_GROUPS["lightweight"])
    return [], []


def _ticker_choices(universe):
    if universe.lower().endswith((".csv", ".txt")):
        path = os.path.expanduser(universe)
        if os.path.exists(path):
            reader = import_module("scripts.run_technical_agents").read_tickers_file
            return reader(path)
        return []
    return [ticker.strip() for ticker in universe.split(",") if ticker.strip()]


def _argument_value(arguments, option, default=""):
    try:
        return arguments[arguments.index(option) + 1]
    except (ValueError, IndexError):
        return default


def _run_scope(command, arguments):
    ticker_source = _argument_value(arguments, "--tickers") or _argument_value(arguments, "--tickers-file")
    try:
        ticker_count = len(_ticker_choices(ticker_source)) if ticker_source else None
    except (OSError, ValueError):
        ticker_count = None
    agents = []
    requested = _argument_value(arguments, "--agents")
    if command in ("technical", "ml") and requested:
        available, _ = _available_agents(command)
        if requested == "all":
            agents = available
        elif command == "ml" and requested in import_module("scripts.run_ml_agents").AGENT_GROUPS:
            agents = list(import_module("scripts.run_ml_agents").AGENT_GROUPS[requested])
        else:
            agents = [name.strip() for name in requested.split(",") if name.strip()]
    return ticker_count, agents


class _QueueWriter:
    def __init__(self, output_queue):
        self.output_queue = output_queue
        self.pending = ""

    def write(self, value):
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.strip():
                self.output_queue.put(line.strip())
        return len(value)

    def flush(self):
        if self.pending.strip():
            self.output_queue.put(self.pending.strip())
        self.pending = ""


def _new_run_state(command, arguments):
    ticker_count, agents = _run_scope(command, arguments)
    return {
        "command": command, "ticker_count": ticker_count, "agents": agents,
        "current_agent": None, "completed_agents": 0, "phase": "Starting runner",
        "outputs": [], "activity": deque(maxlen=7), "elapsed": 0.0,
    }


def update_run_state(state, line):
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(line)).strip()
    if not clean:
        return
    state["activity"].append(clean)
    running = re.match(r"Running\s+(.+?)(?:\.\.\.|$)", clean, re.IGNORECASE)
    if running:
        agent = running.group(1).strip()
        if state["current_agent"] and agent != state["current_agent"]:
            state["completed_agents"] += 1
        state["current_agent"] = agent
        state["phase"] = "Running model" if state["command"] == "ml" else "Calculating signals"
    elif clean.lower().startswith("fetching"):
        state["phase"] = "Fetching market data"
    elif "synthetic" in clean.lower():
        state["phase"] = "Preparing synthetic data"
    elif clean.lower().startswith("model performance"):
        state["phase"] = "Evaluating model"
    elif clean.lower().startswith("wrote "):
        output = clean[6:].strip()
        if output not in state["outputs"]:
            state["outputs"].append(output)
        state["phase"] = "Writing results"
    elif any(word in clean.lower() for word in ("summary", "consensus", "shortlist")):
        state["phase"] = "Building summaries"


def render_run_status(state, finished=False, error=None):
    facts = Table.grid(expand=True, padding=(0, 2))
    facts.add_column(style="dim", width=14)
    facts.add_column(style="bold #d7e4dd")
    facts.add_row("WORKFLOW", state["command"].upper())
    facts.add_row("PHASE", "Failed" if error else ("Complete" if finished else state["phase"]))
    facts.add_row("ELAPSED", f"{state['elapsed']:.1f}s")
    if state["ticker_count"] is not None:
        facts.add_row("TICKERS", str(state["ticker_count"]))
    if state["agents"]:
        current = state["current_agent"] or "waiting"
        done = min(state["completed_agents"], len(state["agents"]))
        if finished and not error:
            done = len(state["agents"])
        bar_width = 20
        filled = round(bar_width * done / len(state["agents"]))
        facts.add_row("AGENT", current)
        facts.add_row("PROGRESS", f"[{'#' * filled}{'.' * (bar_width - filled)}] {done}/{len(state['agents'])}")
    if state["outputs"]:
        facts.add_row("OUTPUTS", "\n".join(state["outputs"][-4:]))
    activity = "\n".join(state["activity"]) or "Waiting for runner output..."
    status = "FAILED" if error else ("COMPLETE" if finished else "RUNNING")
    color = "red" if error else ("green" if finished else "#e66f3c")
    body = Group(facts, Text("\nRECENT ACTIVITY", style="bold #f0a06b"), Text(activity, style="#9bada4"))
    if error:
        body = Group(body, Text(f"\n{error}", style="bold red"))
    return Panel(body, title=f"[bold {color}]{status}[/]", border_style=color, padding=(1, 2))


def run_with_status(command, arguments, dispatch_fn=dispatch, console=None, refresh_interval=0.08):
    console = console or Console()
    state = _new_run_state(command, arguments)
    output_queue = queue.Queue()
    result = {}

    def worker():
        writer = _QueueWriter(output_queue)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                result["value"] = dispatch_fn(command, arguments)
        except BaseException as exc:
            result["error"] = exc
        finally:
            writer.flush()

    started = time.monotonic()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    with Live(render_run_status(state), console=console, screen=True, auto_refresh=False) as live:
        while thread.is_alive() or not output_queue.empty():
            try:
                while True:
                    update_run_state(state, output_queue.get_nowait())
            except queue.Empty:
                pass
            state["elapsed"] = time.monotonic() - started
            live.update(render_run_status(state), refresh=True)
            if thread.is_alive():
                time.sleep(refresh_interval)
        thread.join()
    error = result.get("error")
    console.print(render_run_status(state, finished=True, error=error))
    return result.get("value"), error, state


def _configure_tool(command, console, key_reader=None):
    console.clear()
    console.print(Panel.fit(
        f"[bold #e66f3c]{command.upper()}[/] configuration\n[dim]Type B at any prompt to return[/]",
        border_style="#527c6e",
    ))
    if command == "dashboard":
        universe = _prompt(console.input, "Latest manifest directory", ".")
        agents = ""
        synthetic = False
        lookback = "260"
    else:
        default_universe = "my_tickers.txt" if os.path.exists("my_tickers.txt") else "AAPL,MSFT,NVDA"
        universe = _prompt(console.input, "Tickers or ticker-file path", default_universe)
        universe = choose_ticker_universe(universe, console=console, key_reader=key_reader)
        synthetic = _prompt(console.input, "Use synthetic data? y/n", "n").lower() in ("y", "yes")
        lookback = "260" if synthetic else _prompt(console.input, "Lookback business days", "260")
        agents = ""
        if command in ("technical", "ml"):
            available_agents, default_agents = _available_agents(command)
            selected_agents = select_items(
                "TECHNICAL AGENTS" if command == "technical" else "ML AGENTS",
                available_agents,
                initially_selected=default_agents,
                console=console,
                key_reader=key_reader,
            )
            if selected_agents is None:
                raise BackRequested
            if not selected_agents:
                console.print("[red]Select at least one agent.[/]")
                raise BackRequested
            agents = ",".join(selected_agents)
    extra = _prompt(console.input, "Extra options", "")
    return build_runner_args(command, universe, synthetic, lookback, agents, extra)


def terminal_dashboard(console=None, key_reader=None, dispatch_fn=dispatch):
    console = console or Console()
    key_reader = key_reader or _read_key
    selected = 0
    with Live(render_terminal_dashboard(selected), console=console, screen=True, auto_refresh=False) as live:
        while True:
            live.update(render_terminal_dashboard(selected), refresh=True)
            key = key_reader()
            if key in ("Q", "ESC"):
                return 0
            if key == "LEFT" and selected % 2:
                selected -= 1
            elif key == "RIGHT" and selected % 2 == 0:
                selected += 1
            elif key == "UP" and selected >= 2:
                selected -= 2
            elif key == "DOWN" and selected < 2:
                selected += 2
            elif key == "ENTER":
                command = TOOL_CARDS[selected][0]
                live.stop()
                try:
                    command_args = _configure_tool(command, console, key_reader=key_reader)
                    _, error, _ = run_with_status(
                        command, command_args, dispatch_fn=dispatch_fn, console=console
                    )
                    if error:
                        raise error
                    console.input("\n[dim]Run complete. Press Enter to return to the console...[/]")
                except BackRequested:
                    pass
                except (Exception, SystemExit) as exc:
                    console.print(f"\n[bold red]FAILED[/] {exc}")
                    console.input("\n[dim]Press Enter to return to the console...[/]")
                live.start(refresh=True)

def main(argv=None):
    parser = build_parser()
    args, command_args = parser.parse_known_args(argv)
    if not args.command:
        if argv is None and sys.stdin.isatty():
            return terminal_dashboard()
        parser.print_help()
        return 0
    if args.command.lower() == "menu":
        return terminal_dashboard()
    return dispatch(args.command.lower(), command_args)


if __name__ == "__main__":
    main()
