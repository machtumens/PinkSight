#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use tauri::Manager;

const SIDECAR_ADDR: &str = "127.0.0.1:8756";

struct SidecarState(Mutex<Option<Child>>);

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

fn spawn_sidecar_if_needed() -> Option<Child> {
    if probe_health() {
        return None;
    }
    let sidecar_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("sidecar");
    for interpreter in ["python3", "python"] {
        match Command::new(interpreter)
            .arg("main.py")
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
