"""
setup_and_run.py - Asta AI Launcher
Mengikuti persis urutan manual yang sudah terbukti berjalan.
"""
import sys
import subprocess
import shutil
import time
import webbrowser
from pathlib import Path

ROOT   = Path(__file__).parent.resolve()
UI_DIR = ROOT / "ui/asta-ui"  # nama folder dari: npm create vite@latest asta-ui

# ── Cari venv ─────────────────────────────────────────────────────────────────
def find_venv():
    for name in ("venv", ".venv", "env"):
        p = ROOT / name / "Scripts" / "python.exe"
        if p.exists():
            return ROOT / name
    return None

VENV = find_venv()

def venv_python():
    if VENV:
        return str(VENV / "Scripts" / "python.exe")
    print("[WARN] venv tidak ditemukan, pakai python system")
    return "python"

def venv_uvicorn():
    if VENV:
        uv = VENV / "Scripts" / "uvicorn.exe"
        if uv.exists():
            return str(uv)
    return "uvicorn"

# ─────────────────────────────────────────────────────────────────────────────
def header():
    print()
    print("  ==========================================")
    print("       ASTA AI -- Launcher")
    print("  ==========================================")
    venv_label = str(VENV) if VENV else "tidak ditemukan"
    print(f"  venv   : {venv_label}")
    print(f"  ui dir : {UI_DIR}")
    print()

def check_ui():
    """Pastikan folder asta-ui dan node_modules ada."""
    if not UI_DIR.exists():
        print(f"  [ERROR] Folder '{UI_DIR.name}' tidak ditemukan di {ROOT}")
        print()
        print("  Jalankan dulu secara manual:")
        print("    npm create vite@latest asta-ui -- --template react")
        print("    cd asta-ui && npm install")
        print()
        input("  Tekan Enter untuk keluar...")
        sys.exit(1)

    if not (UI_DIR / "node_modules").exists():
        print(f"  [ERROR] node_modules belum ada di {UI_DIR}")
        print("  Jalankan: cd asta-ui && npm install")
        input("  Tekan Enter untuk keluar...")
        sys.exit(1)

    # Sync file UI terbaru
    src = UI_DIR / "src"
    src.mkdir(exist_ok=True)

    asta_src = ROOT / "AstaUI.jsx"
    if not asta_src.exists():
        print(f"  [ERROR] AstaUI.jsx tidak ada di {ROOT}")
        input("  Tekan Enter untuk keluar...")
        sys.exit(1)

    shutil.copy2(asta_src, src / "AstaUI.jsx")

    # App.jsx — persis seperti yang kamu tulis manual, tanpa import React
    (src / "App.jsx").write_text(
        "import AstaUI from './AstaUI'\n"
        "export default function App() { return <AstaUI /> }\n",
        encoding="utf-8"
    )

    print("  AstaUI.jsx dan App.jsx diperbarui.")

def launch():
    uvicorn = venv_uvicorn()

    # Tulis bat backend — pakai uvicorn dari venv langsung
    backend_bat = ROOT / "_backend.bat"
    if VENV:
        activate = VENV / "Scripts" / "activate.bat"
        backend_bat.write_text(
            f'@echo off\ntitle Asta - Backend\n'
            f'call "{activate}"\n'
            f'cd /d "{ROOT}"\n'
            f'uvicorn api:app --host 0.0.0.0 --port 8000 --reload\n'
            f'pause\n',
            encoding="utf-8"
        )
    else:
        backend_bat.write_text(
            f'@echo off\ntitle Asta - Backend\n'
            f'cd /d "{ROOT}"\n'
            f'uvicorn api:app --host 0.0.0.0 --port 8000 --reload\n'
            f'pause\n',
            encoding="utf-8"
        )

    # Tulis bat frontend — npm run dev di folder asta-ui, tanpa venv
    frontend_bat = ROOT / "_frontend.bat"
    frontend_bat.write_text(
        f'@echo off\ntitle Asta - Frontend\n'
        f'cd /d "{UI_DIR}"\n'
        f'npm run dev\n'
        f'pause\n',
        encoding="utf-8"
    )

    print("  [1/2] Menjalankan backend (venv)...")
    subprocess.Popen(f'start "Asta - Backend" cmd /k "{backend_bat}"', shell=True, cwd=ROOT)

    time.sleep(3)

    print("  [2/2] Menjalankan frontend...")
    subprocess.Popen(f'start "Asta - Frontend" cmd /k "{frontend_bat}"', shell=True, cwd=str(UI_DIR))

    print()
    print("  Menunggu server siap...")
    time.sleep(6)

    print("  Membuka browser --> http://localhost:5173")
    webbrowser.open("http://localhost:5173")

    print()
    print("  Selesai! Tutup window Backend/Frontend untuk berhenti.")
    print()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    header()
    try:
        check_ui()
        launch()
    except KeyboardInterrupt:
        print("\n  Dibatalkan.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    input("  Tekan Enter untuk menutup...\n")