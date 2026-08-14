// Prevents an extra console window on Windows in release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// Minimal Tauri v2 shell. The UI is React; heavy inference is a Python sidecar (see ../sidecar).
// In dev the sidecar is auto-spawned below via a plain std::process::Command (python3 -> python
// fallback), health-probed first to avoid double-launching one, and killed on app exit. If spawn
// fails, the app still loads and the frontend's existing offline -> mock fallback applies — never a
// hard crash. Packaging a bundled sidecar binary via tauri_plugin_shell is a separate follow-on.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use tauri::Manager;

const SIDECAR_ADDR: &str = "127.0.0.1:8756";

// Tracks the auto-spawned sidecar child so it can be killed on app exit. None when a sidecar was
// already running (reused) or when spawn failed.
struct SidecarState(Mutex<Option<Child>>);

// Probe GET /health over a raw TCP socket (no HTTP crate — LOCKED decision 4 forbids a Cargo change).
// Returns true iff a 200 response comes back within the connect/read timeouts; any error -> false.
fn probe_health() -> bool {
    let addr = match SIDECAR_ADDR.parse() {
        Ok(a) => a,
        Err(_) => return false,
    };
    let mut stream = match TcpStream::connect_timeout(&addr, Duration::from_millis(300)) {
        Ok(s) => s,
        Err(_) => return false,
    };
    if stream
        .set_read_timeout(Some(Duration::from_millis(500)))
        .is_err()
    {
        return false;
    }
    let request = b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if stream.write_all(request).is_err() {
        return false;
    }
    let mut buf = [0u8; 64];
    let n = match stream.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return false,
    };
    let head = String::from_utf8_lossy(&buf[..n]);
    head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200")
}

// Spawn the Python sidecar if one is not already healthy. Reuses a running sidecar (returns None, no
// double-launch). The spawn argv ("main.py") and working dir are COMPILE-TIME CONSTANTS — no
// runtime/request-derived string ever reaches Command (LOCKED decision 4). Returns None if both
// interpreters fail (AC10: the app must still load and fall back to offline -> mock).
fn spawn_sidecar_if_needed() -> Option<Child> {
    if probe_health() {
        return None; // already running — reuse, never double-launch
    }
    let sidecar_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("sidecar");
    for interpreter in ["python3", "python"] {
        match Command::new(interpreter)
            .arg("main.py") // fixed literal — never interpolated
            .current_dir(&sidecar_dir)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => return Some(child),
            Err(_) => continue,
        }
    }
    None
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarState(Mutex::new(None)))
        .setup(|app| {
            let child = spawn_sidecar_if_needed();
            if let Some(state) = app.try_state::<SidecarState>() {
                *state.inner().0.lock().unwrap() = child;
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building PinkSight");

    app.run(|app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            if let Some(state) = app_handle.try_state::<SidecarState>() {
                if let Some(mut child) = state.inner().0.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        }
    });
}
