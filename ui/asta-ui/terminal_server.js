import { WebSocketServer } from 'ws';
import { spawn } from 'child_process';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const ROOT_DIR = process.argv[2] 
  || process.env.ROOT_DIR 
  || process.env.PORTABLE_EXECUTABLE_DIR 
  || path.resolve(__dirname, '../../');

const wss = new WebSocketServer({ port: 8001 });
console.log(`[Terminal Server] Root set to: ${ROOT_DIR}`);

let backendProc = null;
let lastTotalTime = 0;
let lastIdleTime = 0;

function getCpuUsage() {
    const cpus = os.cpus();
    let totalTime = 0, idleTime = 0;
    cpus.forEach(cpu => {
        for (let type in cpu.times) totalTime += cpu.times[type];
        idleTime += cpu.times.idle;
    });
    const deltaTotal = totalTime - lastTotalTime;
    const deltaIdle = idleTime - lastIdleTime;
    lastTotalTime = totalTime; lastIdleTime = idleTime;
    return deltaTotal === 0 ? "0.0" : ((1 - deltaIdle / deltaTotal) * 100).toFixed(1);
}

// Menghitung Disk Utilization (% Idle Time terbalik)
function getDiskUtilization() {
    return new Promise((resolve) => {
        const cmd = spawn('wmic', ['path', 'Win32_PerfFormattedData_PerfDisk_LogicalDisk', 'where', 'Name="_Total"', 'get', 'PercentDiskTime', '/format:list'], { shell: true });
        let out = '';
        cmd.stdout.on('data', (d) => out += d.toString());
        cmd.on('close', () => {
            const lines = out.split(/\r?\n/);
            let util = "0.0";
            lines.forEach(l => {
                const parts = l.trim().split('=');
                if (parts[0] === 'PercentDiskTime') util = parts[1];
            });
            // WMIC terkadang mengembalikan nilai > 100 untuk Disk Time jika multi-disk, kita batasi
            const val = parseInt(util) || 0;
            resolve(Math.min(val, 100).toFixed(1));
        });
        cmd.on('error', () => resolve("0.0"));
    });
}

wss.on('connection', (ws) => {
    ws.send(JSON.stringify({ type: 'output', data: `[Asta] Terminal connected.\n[Asta] Root: ${ROOT_DIR}\n` }));

    const shell = spawn('cmd.exe', [], { cwd: ROOT_DIR, env: process.env, shell: true });
    shell.stdout.on('data', (data) => ws.send(JSON.stringify({ type: 'output', data: data.toString() })));
    shell.stderr.on('data', (data) => ws.send(JSON.stringify({ type: 'output', data: data.toString() })));

    const statsInterval = setInterval(async () => {
        if (ws.readyState !== 1) return;
        ws.send(JSON.stringify({ 
            type: 'stats', 
            data: { 
                cpu: getCpuUsage(), 
                ram: (((os.totalmem()-os.freemem())/os.totalmem())*100).toFixed(1), 
                disk: await getDiskUtilization() 
            } 
        }));
    }, 2000);

    ws.on('message', (message) => {
        const input = message.toString().trim();
        const cmd = input.toLowerCase();

        if (cmd === 'cls' || cmd === 'clear') {
            ws.send(JSON.stringify({ type: 'clear' }));
            shell.stdin.write('\n');
            return;
        }

        if (cmd === 'install') {
            ws.send(JSON.stringify({ type: 'output', data: "\n[Asta Setup] Memulai proses instalasi...\n" }));
            shell.stdin.write('py -3.11 -m venv venv || python -m venv venv\n');
            shell.stdin.write('venv\\Scripts\\pip install llama-cpp-python uvicorn fastapi websockets numpy transformers torch sentencepiece accelerate sentence-transformers tavily-python duckduckgo-search huggingface_hub\n');
            return;
        }

        if (cmd === 'start backend') {
            ws.send(JSON.stringify({ type: 'output', data: "[Terminal] Meminta Electron untuk menjalankan backend...\n" }));
            // Sinyal khusus ke frontend agar diteruskan ke IPC
            ws.send(JSON.stringify({ type: 'signal', data: 'start-backend' }));
            return;
        }

        if (cmd === 'stop backend') {
            ws.send(JSON.stringify({ type: 'output', data: "[Terminal] Meminta Electron untuk menghentikan backend...\n" }));
            ws.send(JSON.stringify({ type: 'signal', data: 'stop-backend' }));
            return;
        }

        shell.stdin.write(input + '\n');
    });

    ws.on('close', () => {
        clearInterval(statsInterval);
        shell.kill();
    });
});
