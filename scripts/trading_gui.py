import queue
import shlex
import subprocess
import sys
import threading
from pathlib import Path


COMMAND_AGENT_DEFAULTS = {
    "technical": "all",
    "ml": "lightweight",
}


def build_cli_args(
    command,
    tickers="",
    tickers_file="",
    agents="",
    lookback_days=260,
    use_synthetic=False,
    extra_args="",
):
    if command not in ("quality", "technical", "ml", "dashboard"):
        raise ValueError(f"Unsupported command: {command}")

    args = [sys.executable, "-m", "scripts.trading_cli", command]
    if command == "dashboard":
        if not extra_args.strip():
            args.extend(["--latest-manifest-dir", "."])
    else:
        if tickers_file.strip():
            args.extend(["--tickers-file", tickers_file.strip()])
        elif tickers.strip():
            args.extend(["--tickers", tickers.strip()])
        if use_synthetic:
            args.append("--use-synthetic")
        else:
            args.extend(["--lookback-days", str(int(lookback_days))])
        if command in COMMAND_AGENT_DEFAULTS:
            selected_agents = agents.strip() or COMMAND_AGENT_DEFAULTS[command]
            args.extend(["--agents", selected_agents])

    if extra_args.strip():
        args.extend(shlex.split(extra_args, posix=True))
    return args


class TradingLauncher:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.output_queue = queue.Queue()
        root.title("Trading Tools")
        root.geometry("920x680")
        root.minsize(760, 560)
        root.configure(bg="#f2efe8")

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f2efe8")
        style.configure("TLabel", background="#f2efe8", foreground="#17211d", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Georgia", 24, "bold"), foreground="#163c32")
        style.configure("Hint.TLabel", foreground="#5f6c66")
        style.configure("TButton", font=("Segoe UI Semibold", 10), padding=(14, 8))
        style.configure("Run.TButton", background="#d36b3c", foreground="white")
        style.map("Run.TButton", background=[("active", "#b9532d"), ("disabled", "#c7b5aa")])

        frame = ttk.Frame(root, padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Trading Tools", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            frame,
            text="Run data checks, technical agents, ML agents, or rebuild the dashboard.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(2, 20))

        form = ttk.Frame(frame)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        self.command = tk.StringVar(value="quality")
        self.tickers = tk.StringVar(value="AAPL,MSFT,NVDA")
        self.tickers_file = tk.StringVar()
        self.agents = tk.StringVar()
        self.lookback = tk.StringVar(value="260")
        self.synthetic = tk.BooleanVar(value=True)
        self.extra_args = tk.StringVar()

        self._row(form, 0, "Command", ttk.Combobox(
            form,
            textvariable=self.command,
            values=("quality", "technical", "ml", "dashboard"),
            state="readonly",
        ))
        self._row(form, 1, "Tickers", ttk.Entry(form, textvariable=self.tickers))
        file_frame = ttk.Frame(form)
        file_entry = ttk.Entry(file_frame, textvariable=self.tickers_file)
        file_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(file_frame, text="Browse", command=self._browse).pack(side="left", padx=(8, 0))
        self._row(form, 2, "Ticker file", file_frame)
        self._row(form, 3, "Agents", ttk.Entry(form, textvariable=self.agents))
        self._row(form, 4, "Lookback days", ttk.Entry(form, textvariable=self.lookback))
        self._row(form, 5, "Extra options", ttk.Entry(form, textvariable=self.extra_args))

        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(14, 12))
        ttk.Checkbutton(controls, text="Use synthetic data", variable=self.synthetic).pack(side="left")
        self.run_button = ttk.Button(controls, text="Run", style="Run.TButton", command=self._run)
        self.run_button.pack(side="right")

        self.status = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self.status, style="Hint.TLabel").pack(anchor="w", pady=(0, 6))
        self.output = tk.Text(
            frame,
            height=18,
            wrap="word",
            bg="#13231e",
            fg="#dce8df",
            insertbackground="white",
            relief="flat",
            padx=14,
            pady=12,
            font=("Cascadia Mono", 9),
        )
        self.output.pack(fill="both", expand=True)
        self.command.trace_add("write", self._command_changed)
        self._command_changed()
        self.root.after(100, self._drain_output)

    def _row(self, parent, row, label, widget):
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=5)
        widget.grid(row=row, column=1, sticky="ew", pady=5)

    def _browse(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(filetypes=[("Ticker files", "*.csv *.txt"), ("All files", "*.*")])
        if path:
            self.tickers_file.set(path)

    def _command_changed(self, *_):
        defaults = {"technical": "all", "ml": "lightweight"}
        self.agents.set(defaults.get(self.command.get(), ""))

    def _run(self):
        try:
            command = build_cli_args(
                self.command.get(),
                tickers=self.tickers.get(),
                tickers_file=self.tickers_file.get(),
                agents=self.agents.get(),
                lookback_days=int(self.lookback.get()),
                use_synthetic=self.synthetic.get(),
                extra_args=self.extra_args.get(),
            )
        except (ValueError, TypeError) as exc:
            self.status.set(f"Invalid options: {exc}")
            return

        self.output.delete("1.0", "end")
        self.output.insert("end", f"> {subprocess.list2cmdline(command)}\n\n")
        self.status.set("Running...")
        self.run_button.state(["disabled"])
        threading.Thread(target=self._execute, args=(command,), daemon=True).start()

    def _execute(self, command):
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creation_flags,
            )
            for line in process.stdout:
                self.output_queue.put(("line", line))
            self.output_queue.put(("done", process.wait()))
        except Exception as exc:
            self.output_queue.put(("error", str(exc)))

    def _drain_output(self):
        try:
            while True:
                kind, value = self.output_queue.get_nowait()
                if kind == "line":
                    self.output.insert("end", value)
                    self.output.see("end")
                elif kind == "done":
                    self.status.set("Finished" if value == 0 else f"Failed with exit code {value}")
                    self.run_button.state(["!disabled"])
                else:
                    self.status.set(f"Failed: {value}")
                    self.run_button.state(["!disabled"])
        except queue.Empty:
            pass
        self.root.after(100, self._drain_output)


def main():
    import tkinter as tk

    root = tk.Tk()
    TradingLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
