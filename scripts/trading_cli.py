import argparse
import os
import shlex
import sys
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


def render_selector(title, items, selected, cursor, offset, page_size=12):
    table = Table.grid(expand=True)
    table.add_column(width=4)
    table.add_column(ratio=1)
    visible = items[offset:offset + page_size]
    for row_index, item in enumerate(visible, start=offset):
        active = row_index == cursor
        marker = "[x]" if row_index in selected else "[ ]"
        style = "bold white on #285244" if active else "#c6d6cf"
        table.add_row(marker, Text(str(item), style=style), style=style)
    for _ in range(page_size - len(visible)):
        table.add_row("", "")

    position = f"{cursor + 1}/{len(items)}" if items else "0/0"
    return Panel(
        Group(
            Text(title, style="bold #f0a06b"),
            Text(f"Selected {len(selected)} of {len(items)}", style="dim"),
            table,
            Text("ARROWS scroll   SPACE toggle   A all   N none   ENTER accept   B back", style="bold"),
        ),
        title="[bold]SELECTOR[/]",
        subtitle=position,
        border_style="#527c6e",
        padding=(1, 2),
    )


def select_items(title, items, initially_selected=None, console=None, key_reader=None, page_size=12):
    items = list(dict.fromkeys(items))
    if not items:
        return []
    console = console or Console()
    key_reader = key_reader or _read_key
    initial_values = set(initially_selected) if initially_selected is not None else None
    selected = set(range(len(items))) if initial_values is None else {
        index for index, item in enumerate(items) if item in initial_values
    }
    cursor = 0
    offset = 0
    with Live(
        render_selector(title, items, selected, cursor, offset, page_size),
        console=console,
        screen=True,
        auto_refresh=False,
    ) as live:
        while True:
            live.update(render_selector(title, items, selected, cursor, offset, page_size), refresh=True)
            key = key_reader()
            if key in ("B", "ESC", "Q"):
                return None
            if key == "UP":
                cursor = max(0, cursor - 1)
            elif key == "DOWN":
                cursor = min(len(items) - 1, cursor + 1)
            elif key == "PGUP":
                cursor = max(0, cursor - page_size)
            elif key == "PGDN":
                cursor = min(len(items) - 1, cursor + page_size)
            elif key == "SPACE":
                if cursor in selected:
                    selected.remove(cursor)
                else:
                    selected.add(cursor)
            elif key == "A":
                selected = set(range(len(items)))
            elif key == "N":
                selected.clear()
            elif key == "ENTER":
                return [item for index, item in enumerate(items) if index in selected]
            offset = min(max(0, cursor - page_size + 1), max(0, len(items) - page_size))
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
        ticker_choices = _ticker_choices(universe)
        if ticker_choices:
            chosen_tickers = select_items(
                "TICKERS",
                ticker_choices,
                initially_selected=ticker_choices,
                console=console,
                key_reader=key_reader,
            )
            if chosen_tickers is None:
                raise BackRequested
            if len(chosen_tickers) != len(ticker_choices):
                universe = ",".join(chosen_tickers)
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
                    console.print(f"\n[bold #e66f3c]RUNNING[/] {command} {' '.join(command_args)}\n")
                    dispatch_fn(command, command_args)
                    console.input("\n[dim]Press Enter to return to the console...[/]")
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
