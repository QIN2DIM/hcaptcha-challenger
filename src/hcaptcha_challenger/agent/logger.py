import os
import json
import time
from typing import Any, List, Dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.theme import Theme
from rich.box import ROUNDED
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn

# Force terminal colors if the environment variable is set
force_color = os.getenv("FORCE_COLOR") == "1"

# Initialize Rich Console with a custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "magenta",
    "step": "blue",
    "ai_slow": "bold red",
    "ai_fast": "bold green",
    "network": "cyan"
})

console = Console(theme=custom_theme, force_terminal=force_color if force_color else None)

class NetworkLogger:
    """Aggregates repetitive background task logs into summaries"""
    
    def __init__(self, interval_seconds: float = 5.0):
        self.interval = interval_seconds
        self.request_count = 0
        self.last_log_time = time.time()
        self.active = True

    def log_request(self):
        self.request_count += 1
        now = time.time()
        if now - self.last_log_time >= self.interval:
            duration = now - self.last_log_time
            reqs_per_sec = self.request_count / duration
            # Cleaner network log with no dim style or confusing ANSI sequences
            console.print(
                f"📡 [cyan][NETWORK][/] [bold]{self.request_count}[/] requisições capturadas em [bold]{duration:.1f}s[/] ({reqs_per_sec:.1f} req/s)",
                highlight=False
            )
            self.request_count = 0
            self.last_log_time = now

class ChallengeTracker:
    """Tracks per-round and per-challenge performance metrics"""
    
    def __init__(self):
        self.rounds: List[Dict[str, Any]] = []
        self.start_time = None
        self.challenge_name = "Desafio"

    def start_challenge(self, name: str):
        self.challenge_name = name
        self.start_time = time.time()
        self.rounds = []

    def log_round(self, round_num: int, success: bool, duration: float, ai_time: float, points: int):
        self.rounds.append({
            "round": round_num,
            "success": success,
            "duration": duration,
            "ai_time": ai_time,
            "points": points
        })

    def print_summary(self):
        if not self.rounds:
            return
            
        total_time = time.time() - self.start_time
        ai_total = sum(r['ai_time'] for r in self.rounds)
        success_count = sum(1 for r in self.rounds if r['success'])
        success_rate = (success_count / len(self.rounds)) * 100
        
        table = Table(title=f"📊 RESUMO: {self.challenge_name}", box=ROUNDED, border_style="magenta")
        table.add_column("Round", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Tempo", style="white")
        table.add_column("IA", style="dim")
        table.add_column("Pontos", style="white")
        
        for r in self.rounds:
            status = "[green]✅ ROUND OK[/]" if r['success'] else "[red]❌ FALHA NO ROUND[/]"
            table.add_row(
                str(r['round']),
                status,
                f"{r['duration']:.2f}s",
                f"{r['ai_time']:.2f}s",
                str(r['points'])
            )
            
        console.print(table)
        
        info_text = f"📈 [bold]Total:[/] {total_time:.2f}s | [bold]Sucesso:[/] {success_rate:.1f}% | [bold]IA:[/] {ai_total:.2f}s ({ (ai_total/total_time*100) if total_time > 0 else 0:.1f}%)"
        console.print(Panel(info_text, border_style="magenta", box=ROUNDED))

class LoggerHelper:
    """Helper for beautiful Rich logs with semantic methods"""
    
    EMOJIS = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌',
        'debug': '🔍',
        'start': '🚀',
        'robot': '🤖',
        'human': '🎭',
        'browser': '🌐',
        'mouse': '🖱️',
        'camera': '📸',
        'brain': '🧠',
        'network': '📡',
        'time': '⏱️',
        'flag': '🚩',
        'target': '🎯',
        'refresh': '🔄',
        'check': '✔️',
        'fail': '❌',
        'drag': '🔀',
        'binary': '⚖️',
        'hourglass': '⏳',
        'boom': '💥',
        'trophy': '🏆',
        'skull': '💀',
        'inject': '💉',
        'eye': '👁️',
        'slow': '🐌',
        'fast': '⚡',
        'normal': '🐢'
    }
    
    @staticmethod
    def log_section(title: str, style: str = "bold cyan"):
        """Prints a section header using a Panel"""
        console.print(Panel(f"[{style}]{title}[/]", border_style=style, box=ROUNDED))

    @staticmethod
    def log_info(message: str, emoji: str = "info"):
        """Prints an info message"""
        icon = LoggerHelper.EMOJIS.get(emoji, emoji) or LoggerHelper.EMOJIS['info']
        console.print(f"{icon} {message}", style="info")

    @staticmethod
    def log_warning(message: str, emoji: str = "warning"):
        """Prints a warning message"""
        icon = LoggerHelper.EMOJIS.get(emoji, emoji) or LoggerHelper.EMOJIS['warning']
        console.print(f"{icon} {message}", style="warning")

    @staticmethod
    def log_error(message: str, emoji: str = "error"):
        """Prints an error message"""
        icon = LoggerHelper.EMOJIS.get(emoji, emoji) or LoggerHelper.EMOJIS['error']
        console.print(f"{icon} {message}", style="error")

    @staticmethod
    def log_provider_error(attempt: int, total: int, exception: Exception):
        """Log de erro de provedor (Gemini/Groq) de forma limpa sem JSON verboso"""
        error_msg = str(exception)
        
        # Extrair mensagem principal se for um erro de quota/429
        clean_msg = "Erro desconhecido"
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            clean_msg = "[bold red]Limite de Quota Excedido (429)[/]"
            # Tentar extrair o tempo de espera
            import re
            match = re.search(r"retry in (\d+\.?\d*)s|retryDelay':\s*'(\d+)s'", error_msg)
            if match:
                seconds = match.group(1) or match.group(2)
                clean_msg += f" - Aguarde [yellow]{seconds}s[/]"
        elif "500" in error_msg:
            clean_msg = "[bold yellow]Erro Interno do Servidor (500)[/] - Instabilidade temporária"
        else:
            # Encurtar mensagens genéricas
            clean_msg = error_msg[:100] + "..." if len(error_msg) > 100 else error_msg

        LoggerHelper.log_warning(
            f"Tentativa [bold]{attempt}/{total}[/] - {clean_msg}",
            emoji='warning'
        )

    @staticmethod
    def log_success(message: str, emoji: str = "success"):
        """Prints a success message"""
        icon = LoggerHelper.EMOJIS.get(emoji, emoji) or LoggerHelper.EMOJIS['success']
        console.print(f"{icon} {message}", style="success")

    @staticmethod
    def log_step(step: int, total: int, message: str):
        """Prints a step progress message"""
        percentage = (step / total) * 100
        bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
        console.print(f"[step]🔄 Progresso: {bar} {percentage:.0f}% (Round {step}/{total})[/] - [italic]{message}[/]")

    @staticmethod
    def log_key_value(key: str, value: Any, emoji: str = None):
        """Prints a key-value pair"""
        icon = f"{LoggerHelper.EMOJIS.get(emoji, emoji)} " if emoji else ""
        console.print(f"{icon}[bold]{key}:[/] [highlight]{value}[/]")

    @staticmethod
    def log_json(data: Any, title: str = None):
        """Prints JSON data with syntax highlighting"""
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
        if title:
            console.print(Panel(syntax, title=title, border_style="blue", box=ROUNDED))
        else:
            console.print(syntax)

    # --- Semantic Logging Methods ---

    @staticmethod
    def log_challenge_start(challenge_type: str, round_index: int, total_rounds: int, prompt: str = None, timeout: int = 60):
        """Log semântico para início de desafio com visual premium"""
        content = f"🔀 [bold cyan]Tipo:[/] [yellow]{challenge_type}[/]\n"
        content += f"📝 [bold magenta]Prompt:[/] [italic cyan]\"{prompt[:80]}...\"[/]\n"
        content += f"⏱️  [bold white]Tempo limite:[/] [green]{timeout}s[/]"
        
        console.print(Panel(
            content, 
            title=f"🎯 TENTATIVA {round_index}/{total_rounds}", 
            border_style="cyan", 
            box=ROUNDED,
            padding=(0, 2)
        ))

    @staticmethod
    def log_round_start(current: int, total: int):
        """Log semântico para início de round"""
        LoggerHelper.log_step(current, total, "Iniciando round")

    @staticmethod
    def log_ai_performance(model: str, duration: float, points: int):
        """Log de performance da IA com contexto de velocidade e diagnósticos"""
        if duration > 30:
            status = "ai_slow"
            speed_icon = LoggerHelper.EMOJIS['slow']
            speed_text = "EXTREMAMENTE LENTO"
        elif duration > 15:
            status = "warning"
            speed_icon = LoggerHelper.EMOJIS['normal']
            speed_text = "Lento"
        else:
            status = "ai_fast"
            speed_icon = LoggerHelper.EMOJIS['fast']
            speed_text = "Rápido"
            
        console.print(
            f"  {speed_icon} [{status}]{speed_text} IA ({model}):[/] "
            f"[bold]{duration:.2f}s[/] | "
            f"[bold]Resultado:[/][cyan] {points} pontos encontrados[/]",
            highlight=False
        )
        
        # Alertas de performance lenta solicitados pelo usuário
        if duration > 15:
            LoggerHelper.log_warning(f"IA lenta ({duration:.1f}s). Possíveis causas:", emoji='slow')
            console.print("    • [dim]Rate limit da API ou quota excedida[/]")
            console.print("    • [dim]Modelo sob alta carga (Busy)[/]")
            console.print("    • [dim]Complexidade alta do desafio[/]")
            LoggerHelper.log_info("Tentando otimizar próxima chamada...", emoji='refresh')

    @staticmethod
    def log_mouse_action(action: str, x: int, y: int, element: str = None, duration: float = None):
        """Log de ação do mouse aprimorado"""
        emoji = {
            "click": "🖱️",
            "move": "↗️",
            "drag": "🔀",
            "hover": "👆"
        }.get(action, "⚫")
        
        action_text = action.upper()
        coord_text = f"({x}, {y})"
        elem_text = f" em [bold]{element}[/]" if element else ""
        dur_text = f" em [yellow]{duration:.1f}s[/]" if duration else ""
        
        console.print(f"    {emoji} [dim]{action_text}:[/] {coord_text}{elem_text}{dur_text}", highlight=False)

    @staticmethod
    def log_failure_summary(duration: float, error: str, retry_count: int, total_retries: int = 3):
        """Resumo visual após falha no desafio"""
        action = "Recarregando desafio" if retry_count < total_retries else "Abortando"
        
        content = f"⏱️  [bold white]Tempo gasto:[/] [yellow]{duration:.1f}s[/]\n"
        content += f"🐛 [bold red]Erro:[/] [dim]{error[:100]}...[/]\n"
        content += f"🔄 [bold magenta]Tentativa:[/] [white]{retry_count}/{total_retries}[/]\n"
        content += f"🎯 [bold cyan]Ação:[/] [bold italic]{action}[/]"
        
        console.print(Panel(
            content, 
            title="❌ FALHA NO DESAFIO", 
            border_style="red", 
            box=ROUNDED,
            padding=(0, 2)
        ))


class MetricsLogger:
    """Logger de métricas para estatísticas da sessão"""
    
    def __init__(self):
        self.metrics = {
            "challenges": {"total": 0, "success": 0, "failed": 0},
            "ai_calls": {"total": 0, "total_time": 0.0},
            "errors": {}
        }
        self.start_time = time.time()

    def log_challenge_result(self, success: bool, duration: float):
        self.metrics["challenges"]["total"] += 1
        key = "success" if success else "failed"
        self.metrics["challenges"][key] += 1
        
        status = "✅ SUCESSO" if success else "❌ FALHA"
        color = "green" if success else "red"
        console.print(f"[{color}][bold]{status}[/] em {duration:.2f}s[/]")

    def log_ai_call(self, duration: float):
        self.metrics["ai_calls"]["total"] += 1
        self.metrics["ai_calls"]["total_time"] += duration

    def log_error(self, error_type: str):
        self.metrics["errors"][error_type] = self.metrics["errors"].get(error_type, 0) + 1

    def print_summary(self):
        """Imprime resumo da sessão com Rich Table"""
        duration = time.time() - self.start_time
        
        total_challenges = self.metrics["challenges"]["total"]
        success_rate = (self.metrics["challenges"]["success"] / total_challenges * 100) if total_challenges > 0 else 0
        
        total_ai = self.metrics["ai_calls"]["total"]
        avg_ai = (self.metrics["ai_calls"]["total_time"] / total_ai) if total_ai > 0 else 0
        
        table = Table(title="📊 ESTATÍSTICAS DA SESSÃO", box=ROUNDED, border_style="magenta")
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="bold white")
        
        table.add_row("Tempo Total", f"{duration:.1f}s")
        table.add_row("Desafios", f"{total_challenges} ({success_rate:.1f}% sucesso)")
        table.add_row("Chamadas IA", f"{total_ai} (média: {avg_ai:.2f}s)")
        
        if self.metrics["errors"]:
            error_summary = ", ".join([f"{k}: {v}" for k, v in self.metrics["errors"].items()])
            table.add_row("Erros", f"[red]{error_summary}[/]")
            
        console.print(table)

def log_method_call(emoji: str = None, color: str = 'blue'):
    """Decorador para logar chamadas de métodos importantes"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            instance = args[0] if args else None
            class_name = instance.__class__.__name__ if instance else 'Unknown'
            
            # Log de entrada
            LoggerHelper.log_info(
                f"[bold {color}]{class_name}.{func.__name__}()[/] - Iniciando...",
                emoji=emoji or 'debug'
            )
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # Log de sucesso
                LoggerHelper.log_success(
                    f"[bold {color}]{class_name}.{func.__name__}()[/] - Concluído em {elapsed:.2f}s",
                    emoji='success'
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                
                # Log de erro
                LoggerHelper.log_error(
                    f"[bold {color}]{class_name}.{func.__name__}()[/] - Falhou após {elapsed:.2f}s: {str(e)[:100]}",
                    emoji='error'
                )
                raise
        return async_wrapper
    return decorator

def log_captcha_payload(payload):
    """Log formatado do payload do captcha"""
    if not payload:
        return
    
    table = Table(title="📦 PAYLOAD DO CAPTCHA", box=ROUNDED, border_style="blue")
    table.add_column("Campo", style="bold cyan")
    table.add_column("Valor", style="white")
    
    table.add_row("Tipo", getattr(payload.request_type, 'value', str(payload.request_type)))
    table.add_row("Prompt", f"{payload.get_requester_question()[:50]}...")
    table.add_row("Task ID", str(getattr(payload, 'key', 'N/A')))
    
    if hasattr(payload, 'request_config'):
        config = payload.request_config
        if hasattr(config, 'max_shapes_per_image'):
            table.add_row("Máx formas", str(config.max_shapes_per_image))
            
    console.print(table)
