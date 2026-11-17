# -*- coding: utf-8 -*-
import sqlite3
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger
from hcaptcha_challenger.agent.logger import LoggerHelper

class QuotaManager:
    """
    Manages API key quotas using SQLite for concurrency safety.
    Tracks exhausted (429) and unstable keys.
    Resets daily at 05:00 BRT (08:00 UTC).
    """
    def __init__(self, cache_dir: Path = Path("tmp/.cache")):
        self.db_path = cache_dir.joinpath("quota_manager.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._check_reset()

    def _get_connection(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quotas (
                    key_id TEXT PRIMARY KEY,
                    exhausted_at TEXT,
                    failure_count INTEGER DEFAULT 0,
                    last_failure TEXT,
                    temp_exhausted_until TEXT,
                    backoff_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.commit()

    def _generate_key_id(self, api_key: str, model: str) -> str:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:12]
        return f"{key_hash}_{model}"

    def _check_reset(self):
        """Reset quotas if it's past 08:00 UTC (05:00 BRT)."""
        now_utc = datetime.now(timezone.utc)
        reset_time_today = now_utc.replace(hour=8, minute=0, second=0, microsecond=0)
        
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT value FROM metadata WHERE key = 'last_reset'")
            row = cursor.fetchone()
            last_reset_str = row[0] if row else None
            
            should_reset = False
            if not last_reset_str:
                should_reset = True
            else:
                last_reset = datetime.fromisoformat(last_reset_str)
                if now_utc >= reset_time_today and last_reset < reset_time_today:
                    should_reset = True
            
            if should_reset:
                logger.debug("Reiniciando quotas de chaves API (Reset Diário às 05:00 BRT / 08:00 UTC)")
                conn.execute("DELETE FROM quotas")
                conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_reset', ?)", (now_utc.isoformat(),))
                conn.commit()

    def is_exhausted(self, api_key: str, model: str) -> bool:
        self._check_reset()
        key_id = self._generate_key_id(api_key, model)
        
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT exhausted_at, failure_count, temp_exhausted_until FROM quotas WHERE key_id = ?", (key_id,))
            row = cursor.fetchone()
            if row:
                exhausted_at, failure_count, temp_exhausted_until = row
                
                # 1. Check daily exhaustion
                if exhausted_at:
                    return True
                
                # 2. Check temporary exhaustion (cooldown)
                if temp_exhausted_until:
                    until_dt = datetime.fromisoformat(temp_exhausted_until)
                    if datetime.now(timezone.utc) < until_dt:
                        return True
                    else:
                        # Cooldown expired, clear it
                        conn.execute("UPDATE quotas SET temp_exhausted_until = NULL WHERE key_id = ?", (key_id,))
                        conn.commit()

                # 3. Check instability
                if failure_count and failure_count >= 3:
                    return True
        return False

    def mark_exhausted(self, api_key: str, model: str):
        """Marca uma chave como esgotada com backoff exponencial."""
        key_id = self._generate_key_id(api_key, model)
        
        with self._get_connection() as conn:
            # Obter contador atual
            cursor = conn.execute("SELECT backoff_count FROM quotas WHERE key_id = ?", (key_id,))
            row = cursor.fetchone()
            current_count = row[0] if row else 0
            new_count = current_count + 1
            
            # Backoff: 30s * 2^(n-1) -> 30, 60, 120, 240... max 960 (16min)
            backoff_seconds = min(30 * (2 ** (new_count - 1)), 960)
            until = (datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)).isoformat()
            
            conn.execute("""
                INSERT INTO quotas (key_id, temp_exhausted_until, backoff_count) 
                VALUES (?, ?, ?)
                ON CONFLICT(key_id) DO UPDATE SET 
                    temp_exhausted_until = excluded.temp_exhausted_until,
                    backoff_count = excluded.backoff_count
            """, (key_id, until, new_count))
            conn.commit()
            
        LoggerHelper.log_warning(
            f"Chave [[highlight]{key_id}[/]] esgotada por {backoff_seconds}s (Tentativa {new_count})", 
            emoji='💸'
        )

    def mark_temporary_exhaustion(self, api_key: str, model: str, seconds: int):
        """Mark a key as exhausted for a specific duration (cooldown)."""
        key_id = self._generate_key_id(api_key, model)
        until = (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO quotas (key_id, temp_exhausted_until) 
                VALUES (?, ?)
                ON CONFLICT(key_id) DO UPDATE SET temp_exhausted_until = excluded.temp_exhausted_until
            """, (key_id, until))
            conn.commit()
        LoggerHelper.log_info(f"Chave [[highlight]{key_id}[/]] marcada como ESGOTAMENTO TEMPORÁRIO por [bold]{seconds}s[/]", emoji='hourglass')

    def mark_failure(self, api_key: str, model: str):
        """Track non-429 failures to detect unstable keys."""
        key_id = self._generate_key_id(api_key, model)
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO quotas (key_id, failure_count, last_failure) 
                VALUES (?, 1, ?)
                ON CONFLICT(key_id) DO UPDATE SET 
                    failure_count = failure_count + 1,
                    last_failure = excluded.last_failure
            """, (key_id, now))
            conn.commit()
            
            # Check if it just became unstable
            cursor = conn.execute("SELECT failure_count FROM quotas WHERE key_id = ?", (key_id,))
            count = cursor.fetchone()[0]
            if count >= 3:
                LoggerHelper.log_error(f"Chave [[highlight]{key_id}[/]] marcada como INSTÁVEL após {count} falhas", emoji='boom')

    def mark_success(self, api_key: str, model: str):
        """Reseta falhas e reduz backoff após sucesso."""
        key_id = self._generate_key_id(api_key, model)
        with self._get_connection() as conn:
            # Reduzir backoff count em 2 para recuperação rápida, mas gradual
            conn.execute("""
                UPDATE quotas SET 
                    failure_count = 0,
                    last_failure = NULL,
                    temp_exhausted_until = NULL,
                    backoff_count = MAX(0, backoff_count - 2)
                WHERE key_id = ?
            """, (key_id,))
            conn.commit()
