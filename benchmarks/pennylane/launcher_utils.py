#!/usr/bin/env python3
"""
Cross-platform launcher utilities for benchmarks
Handles launching processes in separate terminals on Windows, Linux, and macOS
"""

import os
import sys
import platform
import subprocess


def launch_in_new_terminal(command, title="Process"):
    """
    Launch a command in a new terminal window (cross-platform)
    
    Args:
        command: Command string to execute
        title: Window title (used on some platforms)
    
    Returns:
        subprocess.Popen object
    """
    system = platform.system()
    
    if system == "Windows":
        # Windows: Use PowerShell with CREATE_NEW_CONSOLE
        return subprocess.Popen(
            ["powershell", "-NoExit", "-Command", command],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    
    elif system == "Linux":
        # Linux: Try common terminal emulators in order of preference
        terminals = [
            # GNOME Terminal
            ["gnome-terminal", "--", "bash", "-c", f"{command}; exec bash"],
            # KDE Konsole
            ["konsole", "-e", "bash", "-c", f"{command}; exec bash"],
            # xterm (fallback)
            ["xterm", "-hold", "-e", command],
            # If in tmux, create new window
            ["tmux", "new-window", command],
        ]
        
        for terminal_cmd in terminals:
            try:
                # Check if terminal is available
                if subprocess.run(["which", terminal_cmd[0]], 
                                capture_output=True).returncode == 0:
                    return subprocess.Popen(terminal_cmd)
            except (FileNotFoundError, PermissionError):
                continue
        
        # Fallback: Run in background without terminal
        print(f"⚠️  No terminal emulator found. Running in background...")
        return subprocess.Popen(command, shell=True)
    
    elif system == "Darwin":  # macOS
        # macOS: Use AppleScript to open Terminal.app
        applescript = f'''
        tell application "Terminal"
            do script "{command}"
            activate
        end tell
        '''
        return subprocess.Popen(["osascript", "-e", applescript])
    
    else:
        # Unknown platform: Run in background
        print(f"⚠️  Unknown platform '{system}'. Running in background...")
        return subprocess.Popen(command, shell=True)


def get_terminal_name():
    """Get friendly name of terminal type for current platform"""
    system = platform.system()
    if system == "Windows":
        return "PowerShell"
    elif system == "Linux":
        return "terminal"
    elif system == "Darwin":
        return "Terminal.app"
    else:
        return "terminal"


def format_command_for_platform(script_path, *args):
    """
    Format a Python command for the current platform
    
    Args:
        script_path: Path to Python script
        *args: Additional arguments to pass to script
    
    Returns:
        Command string formatted for current platform
    """
    system = platform.system()
    script_dir = os.path.dirname(script_path)
    
    # Build argument string
    arg_str = " ".join(f'"{arg}"' for arg in args)
    
    if system == "Windows":
        return f'cd "{script_dir}"; python "{script_path}" {arg_str}'
    else:
        # Linux/macOS
        return f'cd "{script_dir}" && python "{script_path}" {arg_str}'
