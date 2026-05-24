# 
# Copyright (c) 2026 Lorenzo Abate <lorenzo.abate@unina.it>.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
from io import StringIO
from pathlib import Path
import json
import threading
import traceback

from Code_Gen.Kernels import SIMT_RF as simt
from Code_Gen.Kernels import SIMT_XGBoost as simtxgb
from Code_Gen.Kernels import DT_Rec_RF as dtreccodegen
from Code_Gen.Kernels import DT_Rec_XGBoost as dtrecxgb
from Code_Gen.Kernels import FAST_RF as fastrf
from Code_Gen.Kernels import FAST_XGBoost as fastxgb
from Code_Gen.Kernels import HybridL_RF as final
from Code_Gen.Kernels import HybridL_XGBoost as finalxgb
from Code_Gen.Kernels import HybridE_RF as finalenergy
from Code_Gen.Kernels import HybridE_XGBoost as finalenergyxgb

APP_TITLE = "The Code Generator"
RECENT_FILE = Path.home() / ".csv_gui_recent.json"
RECENTS_LIMIT = 10

# ---------- Thread --------------------
def on_simt_done(success: bool, error: str = ""):
    simt_button.config(state="normal")

    if success:
        print("SIMT Generation Completed.")
    else:
        messagebox.showerror("Generation Error", error or "Unknown Error")


def start_simt_thread():
    simt_button.config(state="disabled")

    try:
        csv_path = Path(csv_var.get()).expanduser().resolve()
        n_trees  = int(n_estimators_var.get())
        max_d    = int(max_depth_var.get())      
        seed     = int(random_state_var.get())
        parallel = int(parallelism_var.get())      
        tsize    = int(test_size_var.get())
        
    except Exception as e:
        simt_button.config(state="normal")
        messagebox.showerror("Invalid Parameters", str(e))
        return
    
    model = combo.get()

    def worker():
        try:
            if(model == "Random Forest"):
                print("Starting SIMT generation with Random Forest...")
                simt.generate_simt(csv_path, n_trees, max_d, seed, parallel, tsize)
                root.after(0, on_simt_done, True, "")
                
            elif(model == "XGBoost"):
                print("Starting SIMT generation with XGBoost...")
                simtxgb.generate_simt(csv_path, n_trees, seed, parallel, tsize)
                root.after(0, on_simt_done, True, "")
                
        except Exception as ex:
            err_text = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            root.after(0, on_simt_done, False, err_text)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

def on_dtrec_done(success: bool, error: str = ""):
    dtrec_button.config(state="normal")

    if success:
        print("DT Rec Generation Completed.")
    else:
        messagebox.showerror("Generation Error", error or "Unknown Error")

def start_dtrec_thread():
    dtrec_button.config(state="disabled")

    try:
        csv_path = Path(csv_var.get()).expanduser().resolve()
        n_trees  = int(n_estimators_var.get())
        max_d    = int(max_depth_var.get())      
        seed     = int(random_state_var.get())
        tsize    = int(test_size_var.get())
        
    except Exception as e:
        dtrec_button.config(state="normal")
        messagebox.showerror("Invalid Parameters", str(e))
        return

    model = combo.get()

    def dt_rec_worker():
        try:
            if(model == "Random Forest"):
                print("Starting DT Rec generation with Random Forest...")
                dtreccodegen.generate_dtrec(csv_path, n_trees, max_d, seed, tsize)
                root.after(0, on_dtrec_done, True, "")
            elif(model == "XGBoost"):
                print("Starting DT Rec generation with XGBoost...")
                dtrecxgb.generate_dtrec(csv_path, n_trees, seed, tsize)
                root.after(0, on_dtrec_done, True, "")
        except Exception as ex:
            err_text = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            root.after(0, on_dtrec_done, False, err_text)

    t = threading.Thread(target=dt_rec_worker, daemon=True)
    t.start()

def on_fast_done(success: bool, error: str = ""):
    fast_button.config(state="normal")

    if success:
        print("FAST Generation Completed.")
    else:
        messagebox.showerror("Generation Error", error or "Unknown Error")

def start_fast_thread():
    fast_button.config(state="disabled")

    try:
        csv_path = Path(csv_var.get()).expanduser().resolve()
        n_trees  = int(n_estimators_var.get())
        max_d    = int(max_depth_var.get())      
        seed     = int(random_state_var.get())
        tsize    = int(test_size_var.get())
        
    except Exception as e:
        dtrec_button.config(state="normal")
        messagebox.showerror("Invalid Parameters", str(e))
        return

    model = combo.get()

    def fast_worker():
        try:
            if(model == "Random Forest"):
                print("Starting FAST generation with Random Forest...")
                fastrf.generate_fast(csv_path, n_trees, max_d, seed, tsize)
                root.after(0, on_fast_done, True, "")
            elif(model == "XGBoost"):
                print("Starting FAST generation with XGBoost...")
                fastxgb.generate_fast(csv_path, n_trees, seed, tsize)
                root.after(0, on_fast_done, True, "")
        except Exception as ex:
            err_text = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            root.after(0, on_fast_done, False, err_text)

    t = threading.Thread(target=fast_worker, daemon=True)
    t.start()

def on_final_done(success: bool, error: str = ""):
    final_button.config(state="normal")

    if success:
        print("Hybrid Latency Generation Completed.")
    else:
        messagebox.showerror("Generation Error", error or "Unknown Error")

def start_final_thread():
    final_button.config(state="disabled")

    try:
        csv_path = Path(csv_var.get()).expanduser().resolve()
        n_trees  = int(n_estimators_var.get())
        max_d    = int(max_depth_var.get())      
        seed     = int(random_state_var.get())
        parallel = int(parallelism_var.get())      
        tsize    = int(test_size_var.get())
        
    except Exception as e:
        final_button.config(state="normal")
        messagebox.showerror("Invalid Parameters", str(e))
        return

    model = combo.get()

    def final_worker():
        try:
            if(model == "Random Forest"):
                print("Starting Hybrid Latency generation with Random Forest...")
                final.generate_hybrid_rf(csv_path, n_trees, max_d, seed, parallel, tsize)
                root.after(0, on_final_done, True, "")
            elif(model == "XGBoost"):
                print("Starting Hybrid Latency generation with XGBoost...")
                finalxgb.generate_hybrid_xgb(csv_path, n_trees, max_d, seed, parallel, tsize)
                root.after(0, on_final_done, True, "")

        except Exception as ex:
            err_text = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            root.after(0, on_final_done, False, err_text)

    t = threading.Thread(target=final_worker, daemon=True)
    t.start()

def on_finalenergy_done(success: bool, error: str = ""):
    finalenergy_button.config(state="normal")

    if success:
        print("Hybrid Energy Generation Completed.")
    else:
        messagebox.showerror("Generation Error", error or "Unknown Error")

def start_finalenergy_thread():
    finalenergy_button.config(state="disabled")

    try:
        csv_path = Path(csv_var.get()).expanduser().resolve()
        n_trees  = int(n_estimators_var.get())
        max_d    = int(max_depth_var.get())      
        seed     = int(random_state_var.get())
        parallel = int(parallelism_var.get())      
        tsize    = int(test_size_var.get())
        
    except Exception as e:
        finalenergy_button.config(state="normal")
        messagebox.showerror("Invalid Parameters", str(e))
        return

    model = combo.get()

    def finalenergy_worker():
        try:
            if(model == "Random Forest"):
                print("Starting Hybrid Energy generation with Random Forest...")
                finalenergy.generate_hybrid_rf(csv_path, n_trees, max_d, seed, parallel, tsize)
                root.after(0, on_finalenergy_done, True, "")
            elif(model == "XGBoost"):
                print("Starting Hybrid Energy generation with XGBoost...")
                finalenergyxgb.generate_hybrid_xgb(csv_path, n_trees, max_d, seed, parallel, tsize)
                root.after(0, on_finalenergy_done, True, "")

        except Exception as ex:
            err_text = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            root.after(0, on_finalenergy_done, False, err_text)

    t = threading.Thread(target=finalenergy_worker, daemon=True)
    t.start()


# ---------- Recents ----------
def load_recents():
    try:
        if RECENT_FILE.exists():
            data = json.loads(RECENT_FILE.read_text(encoding="utf-8"))
            # tieni solo path esistenti e normalizzati
            data = [str(Path(p).resolve()) for p in data if Path(p).exists()]
            # de-duplica preservando l'ordine
            seen, out = set(), []
            for p in data:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
            return out[:RECENTS_LIMIT]
    except Exception:
        pass
    return []

def save_recents(recents):
    try:
        RECENT_FILE.write_text(json.dumps(recents[:RECENTS_LIMIT], indent=2), encoding="utf-8")
    except Exception as e:
        messagebox.showwarning("Attention", f"Can't save recents:\n{e}")

def add_recent(path_str):
    p = str(Path(path_str).resolve())
    recents = load_recents()
    if p in recents:
        recents.remove(p)
    recents.insert(0, p)
    save_recents(recents)
    refresh_csv_combo(recents)

def refresh_csv_combo(recents=None):
    if recents is None:
        recents = load_recents()
    csv_combo["values"] = recents

# ---------- Console ----------
class ConsoleOutput(StringIO):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
    def write(self, msg):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, msg)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")
    def flush(self):
        pass

# ---------- Azioni UI ----------
def scegli_csv():
    root.lift()
    root.attributes('-topmost', True)
    root.after(100, lambda: root.attributes('-topmost', False))
    root.focus_force()

    start_dir = Path(csv_var.get()).parent if csv_var.get() else Path.home()
    path = filedialog.askopenfilename(
        parent=root,
        title="CSV Selection",
        initialdir=str(start_dir),
        filetypes=[("CSV File", "*.csv"), ("All Files", "*.*")]
    )

    if path:
        csv_var.set(path)
        add_recent(path)
        print(f"[INFO] Selected CSV: {path}\n")

def on_csv_combo_selected(event=None):
    sel = csv_combo.get().strip()
    if sel:
        csv_var.set(sel)
        # Se esiste, riportiamolo in testa ai recenti
        if Path(sel).exists():
            add_recent(sel)
            print(f"[INFO] Selected from recents {sel}\n")
        else:
            print("[WARN] Selected file doesn't exist anymore\n")

# ---------- UI ----------
root = tk.Tk()
root.title(APP_TITLE)
root.geometry("900x480")

root.lift()
root.attributes('-topmost', True)
root.focus_force()

main = ttk.Frame(root, padding=12)
main.pack(fill="both", expand=True)

# Choose File
csv_var = tk.StringVar()
ttk.Label(main, text="CSV:").grid(row=0, column=0, sticky="w", padx=(0,8), pady=(0,8))

csv_combo = ttk.Combobox(main, textvariable=csv_var, width=60, state="normal")
csv_combo.grid(row=0, column=1, sticky="we", pady=(0,8))
csv_combo.bind("<<ComboboxSelected>>", on_csv_combo_selected)

ttk.Button(main, text="Choose CSV…", command=scegli_csv).grid(row=0, column=2, padx=(8,0), pady=(0,8))

# Parameters
test_size_var = tk.StringVar(value="1000")
n_estimators_var = tk.StringVar(value="32")
random_state_var = tk.StringVar(value="1")
max_depth_var = tk.StringVar(value="10")
parallelism_var = tk.StringVar(value="8")
energy_var = tk.BooleanVar(value=False)

ttk.Label(main, text="Number of Trees:").grid(row=1, column=0, sticky="w", padx=(0,8), pady=4)
ttk.Entry(main, textvariable=n_estimators_var, width=12).grid(row=1, column=1, sticky="w", pady=4)

model_combo = ttk.Frame(main)
model_combo.grid(row=1, column=0, columnspan=3, sticky="e", pady=(12,8))
ttk.Label(model_combo, text="Model:").grid(row=1, column=1, sticky="e", padx=(0,8), pady=4)
combo = ttk.Combobox(model_combo, values=["XGBoost", "Random Forest"], state="readonly", width=14)
combo.grid(row=1, column=2, sticky="e", pady=4)
combo.current(0)

ttk.Label(main, text="Max Depth:").grid(row=2, column=0, sticky="w", padx=(0,8), pady=4)
ttk.Entry(main, textvariable=max_depth_var, width=12).grid(row=2, column=1, sticky="w", pady=4)

ttk.Label(main, text="Random Seed:").grid(row=3, column=0, sticky="w", padx=(0,8), pady=4)
ttk.Entry(main, textvariable=random_state_var, width=12).grid(row=3, column=1, sticky="w", pady=4)

ttk.Label(main, text="Number of Test Samples:").grid(row=4, column=0, sticky="w", padx=(0,8), pady=4)
ttk.Entry(main, textvariable=test_size_var, width=12).grid(row=4, column=1, sticky="w", pady=4)

# Buttons
btns = ttk.Frame(main)
btns.grid(row=4, column=0, columnspan=3, sticky="e", pady=(12,8))
simt_button = ttk.Button(btns, text="SIMT Generation", command=start_simt_thread)
simt_button.pack(side="right")
fast_button = ttk.Button(btns, text="FAST Generation", command=start_fast_thread)
fast_button.pack(side="right", padx=(0,8))
dtrec_button = ttk.Button(btns, text="DT Rec Generation", command=start_dtrec_thread)
dtrec_button.pack(side="right", padx=(0,8))
finalenergy_button = ttk.Button(btns, text="HybridE Generation", command=start_finalenergy_thread)
finalenergy_button.pack(side="right", padx=(0,8))
final_button = ttk.Button(btns, text="HybridL Generation", command=start_final_thread)
final_button.pack(side="right", padx=(0,8))

# Separator
ttk.Separator(main, orient="horizontal").grid(row=6, column=0, columnspan=3, sticky="we", pady=8)

# Console
ttk.Label(main, text="Console output:").grid(row=7, column=0, sticky="w")
console_text = tk.Text(main, height=12, state="disabled", bg="#111", fg="#0f0", insertbackground="white")
console_text.grid(row=8, column=0, columnspan=3, sticky="nsew")

scrollbar = ttk.Scrollbar(main, command=console_text.yview)
console_text.configure(yscrollcommand=scrollbar.set)
scrollbar.grid(row=8, column=3, sticky="ns")

main.columnconfigure(1, weight=1)
main.rowconfigure(8, weight=1)

sys.stdout = ConsoleOutput(console_text)

refresh_csv_combo()
if RECENT_FILE.exists():
    print(f"[INFO] Recents loaded from: {RECENT_FILE}\n")

root.mainloop()
