import json
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Add root to sys.path to allow imports from Utils and Kernels
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import Kernels.DT_Rec_Gen as dtrec_gen
import Kernels.SIMT_Gen as simt_gen
import Utils.trainRF as trainRF
import Utils.trainXGB as trainXGB

RECENT_FILE = Path.home() / ".hybrid_approach_recents.json"
RECENTS_LIMIT = 10


class HybridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid Approach Code Generator")
        self.root.geometry("800x700")

        # Colors from original project
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.console_bg = "#111111"
        self.console_fg = "#00ff00"

        self.root.configure(bg=self.bg_color)
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure(
            "TLabel", background=self.bg_color, foreground=self.fg_color
        )
        self.style.configure(
            "TLabelframe", background=self.bg_color, foreground=self.fg_color
        )
        self.style.configure(
            "TLabelframe.Label", background=self.bg_color, foreground=self.fg_color
        )
        self.style.configure("TButton", background="#444444", foreground=self.fg_color)
        self.style.map("TButton", background=[("active", "#666666")])

        # Variables
        self.csv_var = tk.StringVar()
        self.n_estimators_var = tk.StringVar(value="32")
        self.max_depth_var = tk.StringVar(value="10")
        self.random_state_var = tk.StringVar(value="42")
        self.parallelism_var = tk.StringVar(value="8")
        self.test_size_var = tk.StringVar(value="1000")
        self.model_type_var = tk.StringVar(value="XGBoost")

        self.recents = self.load_recents()
        self.setup_ui()
        self.refresh_csv_combo()

    def load_recents(self):
        if RECENT_FILE.exists():
            try:
                with open(RECENT_FILE, "r") as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_recents(self):
        try:
            with open(RECENT_FILE, "w") as f:
                json.dump(self.recents, f)
        except:
            pass

    def add_recent(self, path):
        path = str(Path(path).resolve())
        if path in self.recents:
            self.recents.remove(path)
        self.recents.insert(0, path)
        self.recents = self.recents[:RECENTS_LIMIT]
        self.save_recents()
        self.refresh_csv_combo()

    def refresh_csv_combo(self):
        self.csv_combo["values"] = self.recents

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)

        # Dataset Selection
        ttk.Label(main_frame, text="Dataset CSV:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.csv_combo = ttk.Combobox(main_frame, textvariable=self.csv_var, width=60)
        self.csv_combo.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.csv_combo.bind(
            "<<ComboboxSelected>>", lambda e: self.add_recent(self.csv_var.get())
        )

        ttk.Button(main_frame, text="Browse", command=self.browse_csv).grid(
            row=0, column=2
        )

        # Parameters
        param_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="15")
        param_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=15)
        for i in range(6):
            param_frame.columnconfigure(i, weight=1)

        params = [
            ("N. Estimators:", self.n_estimators_var),
            ("Max Depth:", self.max_depth_var),
            ("Random State:", self.random_state_var),
            ("Parallelism:", self.parallelism_var),
            ("Test Size:", self.test_size_var),
        ]

        for i, (label, var) in enumerate(params):
            ttk.Label(param_frame, text=label).grid(
                row=i // 3, column=(i % 3) * 2, sticky=tk.W, padx=5, pady=5
            )
            ttk.Entry(param_frame, textvariable=var, width=12).grid(
                row=i // 3, column=(i % 3) * 2 + 1, sticky=tk.EW, padx=5, pady=5
            )

        # Model Selection
        ttk.Label(main_frame, text="Model Type:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        ttk.Combobox(
            main_frame,
            textvariable=self.model_type_var,
            values=["Random Forest", "XGBoost"],
            state="readonly",
            width=20,
        ).grid(row=2, column=1, sticky=tk.W, padx=5)

        # Action Buttons
        button_frame = ttk.Frame(main_frame, padding="10")
        button_frame.grid(row=3, column=0, columnspan=3, pady=20)

        self.simt_btn = ttk.Button(
            button_frame, text="SIMT", command=lambda: self.run_action("SIMT")
        )
        self.simt_btn.grid(row=0, column=0, padx=10)

        self.dtrec_btn = ttk.Button(
            button_frame, text="DT REC", command=lambda: self.run_action("DT_REC")
        )
        self.dtrec_btn.grid(row=0, column=1, padx=10)

        self.fast_btn = ttk.Button(
            button_frame, text="FAST", command=lambda: self.run_action("FAST")
        )
        self.fast_btn.grid(row=0, column=2, padx=10)

        self.hybrid_btn = ttk.Button(
            button_frame, text="HYBRID", command=lambda: self.run_action("HYBRID")
        )
        self.hybrid_btn.grid(row=0, column=3, padx=10)

        # Console Output
        ttk.Label(main_frame, text="Console Output:").grid(row=4, column=0, sticky=tk.W)
        self.console = tk.Text(
            main_frame,
            height=18,
            state="disabled",
            bg=self.console_bg,
            fg=self.console_fg,
            insertbackground="white",
            font=("Consolas", 10),
        )
        self.console.grid(row=5, column=0, columnspan=3, sticky=tk.NSEW, pady=(5, 0))

        scrollbar = ttk.Scrollbar(main_frame, command=self.console.yview)
        self.console.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=5, column=3, sticky="ns", pady=(5, 0))

    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.csv_var.set(path)
            self.add_recent(path)

    def log(self, message):
        self.console.config(state="normal")
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.console.config(state="disabled")
        self.root.update_idletasks()

    def run_action(self, action_name):
        csv_path = self.csv_var.get()
        if not csv_path or not Path(csv_path).exists():
            messagebox.showerror("Error", "Please select a valid CSV file.")
            return

        thread = threading.Thread(target=self.worker, args=(action_name,), daemon=True)
        thread.start()

    def worker(self, action_name):
        try:
            csv_path = self.csv_var.get()
            n_estimators = int(self.n_estimators_var.get())
            max_depth = int(self.max_depth_var.get())
            random_state = int(self.random_state_var.get())
            test_size = int(self.test_size_var.get())
            parallelism = int(self.parallelism_var.get())
            model_type = self.model_type_var.get()

            self.log(f"--- Starting {action_name} Generation ---")
            self.log(f"Training {model_type} model...")

            if model_type == "Random Forest":
                model, task, joblib_path, X_test, acc, mae = trainRF.training(
                    csv_path, n_estimators, max_depth, random_state, test_size
                )
            else:
                model, task, joblib_path, X_test, acc, mae = trainXGB.training(
                    csv_path, n_estimators, max_depth, random_state, test_size
                )

            self.log(f"Model trained and saved to: {joblib_path}")
            self.log(
                f"Task detected: {'Classification' if task == 0 else 'Regression'}"
            )
            if task == 0:
                self.log(f"Model Accuracy: {acc:.4f}")
            else:
                self.log(f"Model MAE: {mae:.4f}")

            if action_name == "SIMT":
                weight = simt_gen.generate_simt(
                    model, task, csv_path, X_test, parallelism
                )
                self.log(f"SIMT Generation complete. Weight: {weight} bytes")
            elif action_name == "DT_REC":
                weight = dtrec_gen.generate_dtrec(model, task, csv_path, X_test)
                self.log(f"DT_REC Generation complete. Weight: {weight} bytes")
            elif action_name == "FAST":
                self.log("FAST Kernel generation not yet implemented.")
            elif action_name == "HYBRID":
                self.log("HYBRID Kernel generation not yet implemented.")

            self.log(f"--- {action_name} Finished ---")

        except Exception as e:
            self.log(f"CRITICAL ERROR: {str(e)}")
            self.log(traceback.format_exc())
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = HybridApp(root)
    root.mainloop()
