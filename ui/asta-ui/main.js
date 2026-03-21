import { app, BrowserWindow, Menu, nativeTheme, ipcMain } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import { fork } from 'child_process';
import fs from 'fs';
import kill from 'tree-kill';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isDev = !app.isPackaged;

let mainWindow;
let terminalServer;
let backendProcess;

const ROOT_DIR = isDev 
  ? path.resolve(__dirname, '../../') 
  : (process.env.PORTABLE_EXECUTABLE_DIR || path.dirname(app.getPath('exe')));

const THEMES = {
  light: { bg: '#f7f6f4', text: '#1a1816' },
  dark:  { bg: '#141210', text: '#e8e0d5' }
};

function updateTitleBar(isDark) {
  if (!mainWindow) return;
  const theme = isDark ? THEMES.dark : THEMES.light;
  mainWindow.setTitleBarOverlay({
    color: theme.bg,
    symbolColor: theme.text,
    height: 35
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    title: "Asta Neural",
    icon: path.join(__dirname, 'build', 'icon.ico'),
    titleBarStyle: 'hidden',
    titleBarOverlay: {
        color: nativeTheme.shouldUseDarkColors ? THEMES.dark.bg : THEMES.light.bg,
        symbolColor: nativeTheme.shouldUseDarkColors ? THEMES.dark.text : THEMES.light.text,
        height: 35
    },
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  Menu.setApplicationMenu(null);

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
  } else {
    mainWindow.loadFile(path.join(__dirname, 'dist', 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

ipcMain.on('theme-changed', (event, mode) => {
  updateTitleBar(mode === 'dark');
});

function startTerminalServer() {
  const terminalPath = isDev 
    ? path.join(__dirname, 'terminal_server.js')
    : path.join(process.resourcesPath, 'app.asar.unpacked', 'terminal_server.js');
    
  console.log(`[Main] Starting terminal from: ${terminalPath}`);
  
  terminalServer = fork(terminalPath, [ROOT_DIR], {
    env: { ...process.env, ROOT_DIR }
  });
}

function startBackend() {
  if (backendProcess) return;

  const pythonPath = path.join(ROOT_DIR, 'venv', 'Scripts', 'python.exe');

  if (!fs.existsSync(pythonPath)) {
    console.error("Backend failed: Python executable not found at", pythonPath);
    return;
  }

  // Tambahkan --reload jika ingin auto-restart, tapi uvicorn --reload akan buat sub-proses baru
  const args = ['-m', 'uvicorn', 'api:app', '--host', '0.0.0.0', '--port', '8000'];

  backendProcess = spawn(pythonPath, args, {
    cwd: ROOT_DIR,
    shell: false,
    env: { ...process.env, PYTHONUNBUFFERED: "1" }
  });

  backendProcess.stdout.on('data', (data) => {
    const msg = data.toString();
    console.log(`[Backend] ${msg}`);
    // Kirim ke Terminal via IPC jika ada jendela terbuka
    if (mainWindow) {
        mainWindow.webContents.send('backend-out', msg);
    }
  });

  backendProcess.stderr.on('data', (data) => {
    const msg = data.toString();
    console.error(`[Backend] ${msg}`);
    if (mainWindow) {
        mainWindow.webContents.send('backend-err', msg);
    }
  });

  backendProcess.on('close', (code) => {
    console.log(`Backend process closed with code ${code}`);
    backendProcess = null;
    if (mainWindow) {
        mainWindow.webContents.send('backend-status', 'stopped');
    }
  });
}

function stopBackend() {
    if (backendProcess && backendProcess.pid) {
        kill(backendProcess.pid, 'SIGKILL');
        backendProcess = null;
    }
}

ipcMain.on('start-backend', () => startBackend());
ipcMain.on('stop-backend', () => stopBackend());
ipcMain.on('restart-backend', () => {
    console.log("[Main] Restarting backend for device/config change...");
    stopBackend();
    // Beri jeda 1.5 detik agar port 8000 benar-benar lepas dan VRAM/RAM bersih
    setTimeout(() => {
        startBackend();
    }, 1500);
});

ipcMain.on('theme-changed', (event, mode) => {
  updateTitleBar(mode === 'dark');
});

function killProcesses() {
    if (backendProcess && backendProcess.pid) {
        console.log("Killing Backend Tree:", backendProcess.pid);
        kill(backendProcess.pid, 'SIGKILL');
        backendProcess = null;
    }
    if (terminalServer) {
        terminalServer.kill();
    }
}

app.whenReady().then(() => {
  startTerminalServer();
  createWindow();

  nativeTheme.on('updated', () => {
    updateTitleBar(nativeTheme.shouldUseDarkColors);
  });
});

app.on('before-quit', () => {
    killProcesses();
});

app.on('window-all-closed', () => {
  killProcesses();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
