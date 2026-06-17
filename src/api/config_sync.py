"""
Configuration Synchronization Service
Handles coordination between FastAPI backend and main application configuration
"""

import asyncio
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.utils.config_loader import ConfigLoader
from src.utils.logger import logger


class ConfigSyncService:
    """
    Service for synchronizing configuration changes between FastAPI backend and main application
    """

    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.sync_file = Path("tmp/config_sync.json")
        self.lock_file = Path("tmp/config_sync.lock")
        self.main_app_pid_file = Path("tmp/main_app.pid")
        self.restart_required_sections = {
            "system",
            "broker",
            "trading",
            "model_training",
            "database",
            "performance",
            "scheduler",
        }

        # Ensure tmp directory exists
        self.sync_file.parent.mkdir(exist_ok=True)

    async def sync_config_update(self, section: str, data: dict[str, Any]) -> dict[str, Any]:
        """
        Synchronize configuration update with main application

        Returns:
            Dict with sync status and actions taken
        """
        try:
            # Check if sync is already in progress
            if self.lock_file.exists():
                raise Exception("Configuration sync already in progress")

            # Create lock file
            self._create_lock()

            try:
                # Update configuration
                backup_path = self.config_loader.backup_config()
                self.config_loader.update_config_section(section, data)

                # Determine if restart is required
                restart_required = section in self.restart_required_sections

                # Create sync notification
                sync_data = {
                    "timestamp": datetime.now().isoformat(),
                    "section": section,
                    "data": data,
                    "restart_required": restart_required,
                    "backup_path": backup_path,
                    "source": "api",
                }

                # Write sync file for main application
                with open(self.sync_file, "w") as f:
                    json.dump(sync_data, f, indent=2)

                # Signal main application if running
                if restart_required:
                    await self._signal_main_app_restart()
                else:
                    await self._signal_main_app_reload()

                return {
                    "success": True,
                    "backup_created": backup_path,
                    "restart_required": restart_required,
                    "sync_file": str(self.sync_file),
                    "action": "restart" if restart_required else "reload",
                }

            finally:
                # Remove lock file
                self._remove_lock()

        except Exception as e:
            logger.error(f"Configuration sync failed: {e}")
            raise

    async def check_main_app_status(self) -> dict[str, Any]:
        """Check if main application is running and responsive"""
        try:
            if not self.main_app_pid_file.exists():
                return {"running": False, "reason": "PID file not found"}

            with open(self.main_app_pid_file) as f:
                pid = int(f.read().strip())

            # Check if process is running
            try:
                os.kill(pid, 0)  # Signal 0 checks if process exists
                return {"running": True, "pid": pid}
            except ProcessLookupError:
                return {"running": False, "reason": "Process not found"}

        except Exception as e:
            logger.error(f"Failed to check main app status: {e}")
            return {"running": False, "reason": str(e)}

    async def _signal_main_app_restart(self) -> None:
        """Signal main application to restart"""
        try:
            status = await self.check_main_app_status()
            if status["running"]:
                pid = status["pid"]
                # Send SIGTERM for graceful shutdown
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Sent restart signal to main application (PID: {pid})")

                # Wait for shutdown with timeout
                await self._wait_for_shutdown(pid, timeout=30)

        except Exception as e:
            logger.error(f"Failed to signal main app restart: {e}")

    async def _signal_main_app_reload(self) -> None:
        """Signal main application to reload configuration"""
        try:
            status = await self.check_main_app_status()
            if status["running"]:
                pid = status["pid"]
                # Send SIGUSR1 for configuration reload
                os.kill(pid, signal.SIGUSR1)
                logger.info(f"Sent config reload signal to main application (PID: {pid})")

        except Exception as e:
            logger.error(f"Failed to signal main app reload: {e}")

    async def _wait_for_shutdown(self, pid: int, timeout: int = 30) -> None:
        """Wait for process to shutdown with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                os.kill(pid, 0)
                await asyncio.sleep(1)
            except ProcessLookupError:
                logger.info(f"Main application (PID: {pid}) shutdown successfully")
                return

        logger.warning(f"Main application (PID: {pid}) did not shutdown within {timeout}s")

    def _create_lock(self) -> None:
        """Create lock file to prevent concurrent sync operations"""
        lock_data = {"timestamp": datetime.now().isoformat(), "api_pid": os.getpid()}

        with open(self.lock_file, "w") as f:
            json.dump(lock_data, f, indent=2)

    def _remove_lock(self) -> None:
        """Remove lock file"""
        if self.lock_file.exists():
            self.lock_file.unlink()

    def get_sync_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get configuration sync history"""
        history_file = Path("tmp/config_sync_history.json")

        if not history_file.exists():
            return []

        try:
            with open(history_file) as f:
                history: list[dict[str, Any]] = json.load(f)

            return history[-limit:] if len(history) > limit else history

        except Exception as e:
            logger.error(f"Failed to read sync history: {e}")
            return []

    def add_sync_history(self, section: str, data: dict[str, Any], success: bool, error: Optional[str] = None) -> None:
        """Add entry to sync history"""
        history_file = Path("tmp/config_sync_history.json")

        entry = {
            "timestamp": datetime.now().isoformat(),
            "section": section,
            "data": data,
            "success": success,
            "error": error,
        }

        try:
            # Load existing history
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
            else:
                history = []

            # Add new entry
            history.append(entry)

            # Keep only last 100 entries
            history = history[-100:]

            # Save updated history
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save sync history: {e}")

    def validate_config_change(self, section: str, data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate configuration change before applying

        Returns:
            Dict with validation results
        """
        try:
            # Use config loader validation
            self.config_loader.validate_config_section(section, data)

            # Additional business logic validations
            validation_warnings = []

            # Check for potentially dangerous changes
            if section == "system" and "mode" in data and data["mode"] != self.config_loader.get_config().system.mode:
                validation_warnings.append("Mode change requires system restart and may affect active operations")

            if section == "broker" and any(key in data for key in ["api_key", "api_secret"]):
                validation_warnings.append("Broker credentials change requires authentication restart")

            if section == "trading" and "instruments" in data:
                validation_warnings.append("Instrument configuration change may affect active trading positions")

            return {
                "valid": True,
                "warnings": validation_warnings,
                "restart_required": section in self.restart_required_sections,
            }

        except Exception as e:
            return {"valid": False, "error": str(e), "restart_required": False}

    def get_config_diff(self, section: str, new_data: dict[str, Any]) -> dict[str, Any]:
        """
        Get diff between current and new configuration

        Returns:
            Dict with changes
        """
        try:
            current_data = self.config_loader.get_config_section(section)

            changes = {}
            for key, new_value in new_data.items():
                current_value = current_data.get(key)
                if current_value != new_value:
                    changes[key] = {"old": current_value, "new": new_value}

            return {"section": section, "changes": changes, "total_changes": len(changes)}

        except Exception as e:
            logger.error(f"Failed to get config diff: {e}")
            return {"section": section, "changes": {}, "total_changes": 0}


# Global service instance
_sync_service: Optional[ConfigSyncService] = None


def get_config_sync_service() -> ConfigSyncService:
    """Get global configuration sync service instance"""
    global _sync_service

    if _sync_service is None:
        config_loader = ConfigLoader()
        _sync_service = ConfigSyncService(config_loader)

    return _sync_service
