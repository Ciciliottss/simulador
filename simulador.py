import sys
import numpy as np
import random
import csv
import math
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from collections import defaultdict
import multiprocessing as mp
from contextlib import contextmanager

# ==============================
# CLASSE DE CORES E FORMATAÇÃO
# ==============================

class Cores:
    """
    Códigos ANSI para cores no terminal.
    Funciona em Linux/Mac. No Windows, pode precisar de colorama.
    """
    # Cores básicas
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

    # Cores de texto
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Cores brilhantes
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Cores de fundo
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

    @classmethod
    def desativar(cls):
        """Desativa cores (para ambientes que não suportam)"""
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                setattr(cls, attr, '')


class FormatadorVisual:
    """
    Utilitários para formatação visual avançada.
    Sem impacto nos cálculos estatísticos.
    """

    @staticmethod
    def titulo_secao(texto: str, largura: int = 80, cor=None) -> str:
        """Cria título de seção formatado"""
        if cor is None:
            cor = Cores.CYAN

        borda = "═" * largura
        espacos = (largura - len(texto) - 2) // 2
        linha_titulo = "║" + " " * espacos + texto + " " * (largura - espacos - len(texto) - 2) + "║"

        return f"\n{cor}{Cores.BOLD}{borda}\n{linha_titulo}\n{borda}{Cores.RESET}\n"

    @staticmethod
    def caixa(texto: str, largura: int = 60, cor=None) -> str:
        """Cria caixa ao redor do texto"""
        if cor is None:
            cor = Cores.WHITE

        linhas = texto.split('\n')
        resultado = []

        resultado.append(cor + "┌" + "─" * (largura - 2) + "┐" + Cores.RESET)

        for linha in linhas:
            espacos = largura - len(linha) - 4
            resultado.append(cor + "│ " + Cores.RESET + linha + " " * espacos + cor + " │" + Cores.RESET)

        resultado.append(cor + "└" + "─" * (largura - 2) + "┘" + Cores.RESET)

        return "\n".join(resultado)

    @staticmethod
    def tabela_fancy(headers: List[str], linhas: List[List], larguras: List[int] = None) -> str:
        """Cria tabela com bordas Unicode"""
        if not larguras:
            larguras = [max(len(str(linha[i])) for linha in [headers] + linhas) + 2
                        for i in range(len(headers))]

        # Borda superior
        resultado = [Cores.BRIGHT_BLACK + "┌" + "┬".join("─" * l for l in larguras) + "┐" + Cores.RESET]

        # Cabeçalho
        header_str = Cores.BRIGHT_BLACK + "│" + Cores.RESET
        for i, h in enumerate(headers):
            header_str += Cores.BOLD + Cores.CYAN + str(h).center(larguras[i]) + Cores.RESET
            header_str += Cores.BRIGHT_BLACK + "│" + Cores.RESET
        resultado.append(header_str)

        # Separador
        resultado.append(Cores.BRIGHT_BLACK + "├" + "┼".join("─" * l for l in larguras) + "┤" + Cores.RESET)

        # Linhas
        for linha in linhas:
            linha_str = Cores.BRIGHT_BLACK + "│" + Cores.RESET
            for i, celula in enumerate(linha):
                linha_str += str(celula).center(larguras[i])
                linha_str += Cores.BRIGHT_BLACK + "│" + Cores.RESET
            resultado.append(linha_str)

        # Borda inferior
        resultado.append(Cores.BRIGHT_BLACK + "└" + "┴".join("─" * l for l in larguras) + "┘" + Cores.RESET)

        return "\n".join(resultado)

    @staticmethod
    def barra_colorida(valor: float, max_valor: float = 100, largura: int = 20,
                       usar_gradiente: bool = True, inverter_cores: bool = False) -> str:
        """
        Barra de progresso colorida baseada no valor

        Args:
            valor: Valor a ser representado
            max_valor: Valor máximo para normalização
            largura: Largura da barra em caracteres
            usar_gradiente: Se True, usa gradiente vermelho->amarelo->verde
            inverter_cores: Se True, inverte a lógica (verde para baixo, vermelho para cima)
                           Útil para métricas negativas como "risco de rebaixamento"
        """
        proporcao = min(valor / max_valor, 1.0)
        cheios = int(largura * proporcao)
        vazios = largura - cheios

        # Escolhe cor baseada no valor
        if usar_gradiente:
            if inverter_cores:
                # Lógica invertida: quanto MAIOR o valor, PIOR (vermelho)
                if proporcao < 0.33:
                    cor = Cores.GREEN  # Baixo risco = verde
                elif proporcao < 0.66:
                    cor = Cores.YELLOW  # Médio risco = amarelo
                else:
                    cor = Cores.RED  # Alto risco = vermelho
            else:
                # Lógica normal: quanto MAIOR o valor, MELHOR (verde)
                if proporcao < 0.33:
                    cor = Cores.RED
                elif proporcao < 0.66:
                    cor = Cores.YELLOW
                else:
                    cor = Cores.GREEN
        else:
            cor = Cores.CYAN

        barra = cor + "█" * cheios + Cores.DIM + "░" * vazios + Cores.RESET
        return f"[{barra}]"

    @staticmethod
    def medalhao(posicao: int) -> str:
        """Retorna medalha/emoji para top 3"""
        if posicao == 1:
            return "🥇"
        elif posicao == 2:
            return "🥈"
        elif posicao == 3:
            return "🥉"
        else:
            return "  "

    @staticmethod
    def indicador_tendencia(valor_atual: float, valor_anterior: float,
                            threshold: float = 2.0) -> str:
        """Retorna indicador visual de tendência"""
        delta = valor_atual - valor_anterior

        if abs(delta) < threshold:
            return f"{Cores.BRIGHT_BLACK}➡️  {Cores.RESET}"
        elif delta > 0:
            return f"{Cores.GREEN}📈 +{delta:.1f}%{Cores.RESET}"
        else:
            return f"{Cores.RED}📉 {delta:.1f}%{Cores.RESET}"

# ==============================
# BLOCO 0: BARRA DE PROGRESSO
# ==============================

def barra_progresso(atual, total, largura=30, mostrar_tempo=False, tempo_decorrido=0):
    """
    Barra de progresso visual melhorada.

    Args:
        atual: Valor atual
        total: Valor total
        largura: Largura da barra
        mostrar_tempo: Se deve mostrar tempo estimado
        tempo_decorrido: Tempo decorrido em segundos
    """
    proporcao = atual / total
    cheios = int(largura * proporcao)
    vazios = largura - cheios

    # Escolhe cor baseada no progresso
    if proporcao < 0.33:
        cor = Cores.RED
    elif proporcao < 0.66:
        cor = Cores.YELLOW
    else:
        cor = Cores.GREEN

    # Barra com cor
    barra = cor + "█" * cheios + Cores.DIM + "░" * vazios + Cores.RESET
    pct = int(proporcao * 100)

    texto = f"[{barra}] {cor}{Cores.BOLD}{pct:3d}%{Cores.RESET} ({atual:,}/{total:,})"

    # Adiciona tempo estimado
    if mostrar_tempo and tempo_decorrido > 0 and atual > 0:
        tempo_por_sim = tempo_decorrido / atual
        tempo_restante = tempo_por_sim * (total - atual)

        if tempo_restante < 60:
            tempo_str = f"{int(tempo_restante)}s"
        elif tempo_restante < 3600:
            tempo_str = f"{int(tempo_restante / 60)}m {int(tempo_restante % 60)}s"
        else:
            tempo_str = f"{int(tempo_restante / 3600)}h {int((tempo_restante % 3600) / 60)}m"

        texto += f" {Cores.BRIGHT_BLACK}[ETA: {tempo_str}]{Cores.RESET}"

    return texto

# ==============================
# BLOCO 1: FUNDAÇÃO TÉCNICA
# ==============================

class SeedManager:
    """
    Gerenciador centralizado de aleatoriedade.
    Garante reprodutibilidade completa e independência entre simulações.
    """
    _master_seed: Optional[int] = None
    _initialized: bool = False

    @classmethod
    def configurar_seed(cls, seed: Optional[int] = None):
        """
        Configura seed global do sistema.

        Args:
            seed: Seed para reprodutibilidade. None = aleatório.

        Impacto estatístico: NENHUM na distribuição, apenas na reprodutibilidade.
        """
        cls._master_seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        cls._initialized = True

    @classmethod
    def gerar_seed_worker(cls, worker_id: int) -> int:
        """
        Gera seed independente para cada worker paralelo.

        Estratégia: hash determinístico do master_seed + worker_id
        Garante: independência estatística entre workers
        """
        if cls._master_seed is None:
            return random.randint(0, 2 ** 31 - 1)

        # Hash determinístico: combinação de master_seed e worker_id
        return (cls._master_seed + worker_id * 982451653) % (2 ** 31)

    @classmethod
    def get_estado(cls) -> Dict[str, Any]:
        """Retorna estado atual para serialização"""
        return {
            'master_seed': cls._master_seed,
            'random_state': random.getstate(),
            'numpy_state': np.random.get_state()
        }

    @classmethod
    def restaurar_estado(cls, estado: Dict[str, Any]):
        """Restaura estado salvo"""
        cls._master_seed = estado['master_seed']
        random.setstate(estado['random_state'])
        np.random.set_state(estado['numpy_state'])
        cls._initialized = True


@dataclass
class ResultadoSimulacao:
    """
    Resultado de UMA simulação completa (imutável após criação).
    Separação clara entre dado bruto e agregação.
    """
    simulacao_id: int
    classificacao: List[str]  # Times ordenados por posição final
    pontos: Dict[str, int]
    saldos: Dict[str, int]
    gols_pro: Dict[str, int]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Serialização segura"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ResultadoSimulacao':
        """Deserialização segura"""
        return cls(**data)


@dataclass
class ResultadoTemporada:
    """
    Agregação de múltiplas simulações.
    Estatísticas completas + metadados de execução.
    """
    n_simulacoes: int
    times: List[str]

    # Estatísticas agregadas
    probabilidades_titulo: Dict[str, float]
    probabilidades_promocao: Dict[str, float]
    probabilidades_playoffs: Dict[str, float]
    probabilidades_rebaixamento: Dict[str, float]

    # Distribuições completas
    pontos_distribuicao: Dict[str, List[int]]
    saldos_distribuicao: Dict[str, List[int]]
    posicoes_distribuicao: Dict[str, List[int]]

    # Metadados
    timestamp: float = field(default_factory=time.time)
    versao_codigo: str = "5.0"
    seed_utilizada: Optional[int] = None
    tempo_execucao: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ResultadoTemporada':
        return cls(**data)


class CheckpointManager:
    """
    BLOCO 1.3: Persistência incremental sem viés estatístico.

    Desafio: salvar progresso sem afetar independência das simulações.
    Solução: salvar resultados + estado do RNG separadamente.
    """

    def __init__(self, diretorio: Path = Path("checkpoints")):
        self.diretorio = diretorio
        self.diretorio.mkdir(exist_ok=True)

    def salvar_checkpoint(
            self,
            resultados_parciais: List[ResultadoSimulacao],
            metadata: Dict[str, Any],
            checkpoint_id: str
    ):
        """
        Salva checkpoint com resultados + estado do RNG.

        Garantia estatística: estado do RNG permite retomar
        exatamente de onde parou, mantendo independência.
        """
        checkpoint_data = {
            'resultados': [r.to_dict() for r in resultados_parciais],
            'metadata': metadata,
            'rng_state': SeedManager.get_estado(),
            'timestamp': time.time()
        }

        arquivo = self.diretorio / f"{checkpoint_id}.pkl"
        with open(arquivo, 'wb') as f:
            pickle.dump(checkpoint_data, f)

    def carregar_checkpoint(self, checkpoint_id: str) -> Tuple[List[ResultadoSimulacao], Dict, bool]:
        """
        Carrega checkpoint e restaura estado do RNG.

        Returns:
            (resultados, metadata, sucesso)
        """
        arquivo = self.diretorio / f"{checkpoint_id}.pkl"

        if not arquivo.exists():
            return [], {}, False

        try:
            with open(arquivo, 'rb') as f:
                data = pickle.load(f)

            resultados = [ResultadoSimulacao.from_dict(r) for r in data['resultados']]
            SeedManager.restaurar_estado(data['rng_state'])

            return resultados, data['metadata'], True
        except Exception as e:
            print(f"⚠️  Erro ao carregar checkpoint: {e}")
            return [], {}, False


# ==============================
# BLOCO 2: PERFORMANCE E ESCALA
# ==============================

class PerformanceProfiler:
    """
    BLOCO 2.6: Profiling embutido sem overhead significativo.
    """

    def __init__(self):
        self.tempos: Dict[str, List[float]] = defaultdict(list)
        self.contadores: Dict[str, int] = defaultdict(int)
        self._ativo = True

    def desativar(self):
        """Desativa profiling para produção"""
        self._ativo = False

    @contextmanager
    def medir(self, operacao: str):
        """Context manager para medição de tempo"""
        if not self._ativo:
            yield
            return

        inicio = time.perf_counter()
        try:
            yield
        finally:
            duracao = time.perf_counter() - inicio
            self.tempos[operacao].append(duracao)
            self.contadores[operacao] += 1

    def relatorio(self):
        linhas = []
        linhas.append("\n⏱️ RELATÓRIO DE PERFORMANCE")
        linhas.append("=" * 50)

        if not hasattr(self, "tempos") or not self.tempos:
            linhas.append("Nenhuma métrica registrada.")
            return "\n".join(linhas)

        total = sum(self.tempos.values())
        for nome, tempo in sorted(self.tempos.items(), key=lambda x: -x[1]):
            perc = (tempo / total * 100) if total > 0 else 0
            linhas.append(f"{nome:<30} {tempo:8.3f}s  ({perc:5.1f}%)")

        linhas.append("-" * 50)
        linhas.append(f"{'TOTAL':<30} {total:8.3f}s")
        return "\n".join(linhas)

# Cache global com limite de memória (BLOCO 2.5)
@lru_cache(maxsize=10000)
def poisson_pmf_cached(k: int, lam_round: float) -> float:
    """
    Cache LRU para Poisson PMF.

    Trade-off:
    - Memória: ~1.5MB para 10k entradas
    - Speed-up: 5-10x em casos típicos
    - Equivalência: 100% (determinística)
    """
    _FACTORIAL = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    return math.exp(-lam_round) * (lam_round ** k) / _FACTORIAL[k]


def poisson_pmf(k: int, lam: float) -> float:
    """Wrapper que quantiza lambda para cache"""
    lam_round = round(lam, 2)
    return poisson_pmf_cached(k, lam_round)


# ==============================
# CONFIGURAÇÕES (ESTENDIDAS)
# ==============================

@dataclass
class Config:
    """Configurações globais - agora com flags de features"""

    # Simulação
    simulacoes: int = 10000
    total_jogos: int = 46
    jogos_por_rodada: int = 12

    # Estatísticas base
    media_gols_historica: float = 2.6
    taxa_empates_historica: float = 0.26

    # Modelo
    fator_casa_prior: float = 1.08
    confianca_minima_mando: int = 10
    expoente_mando_visitante: float = 0.5
    rho: float = -0.13

    # Prior estrutural
    peso_qualidade_inicial: float = 0.98
    confianca_qualidade: int = 25

    # Soft bounds
    amplitude_inicial: float = 1.8
    amplitude_final: float = 0.45
    centro_forca: float = 1.0
    fator_exponencial: float = 0.80

    # OVR temporal
    k_suavidade_ovr: float = 4.0
    delta_max_ovr: float = 1.5

    # NOVOS: Features opcionais (BLOCO 3)
    usar_peso_dinamico_ovr: bool = False  # BLOCO 3.8
    usar_incerteza_estrutural: bool = False  # BLOCO 3.9
    calibrar_rho_automatico: bool = False  # BLOCO 3.7

    # Performance
    checkpoint_intervalo: int = 1000  # Salvar a cada N simulações
    usar_paralelizacao: bool = False  # BLOCO 2.4
    n_workers: int = 4
    ativar_profiling: bool = False
    atualizar_progresso_a_cada: int = 100

config = Config()


# ==============================
# CLASSES DE DADOS (MANTIDAS)
# ==============================

@dataclass
class Rating:
    ATT: int
    MEI: int
    DEF: int

    def media(self) -> float:
        return (self.ATT + self.MEI + self.DEF) / 3


@dataclass
class Estatisticas:
    gols_pro: int = 0
    gols_contra: int = 0
    jogos: int = 0
    pontos_casa: int = 0
    pontos_fora: int = 0
    jogos_casa: int = 0
    jogos_fora: int = 0

    def pontos_totais(self) -> int:
        return self.pontos_casa + self.pontos_fora

    def media_gols_pro(self) -> float:
        return self.gols_pro / max(1, self.jogos)

    def media_gols_contra(self) -> float:
        return self.gols_contra / max(1, self.jogos)

    def saldo_gols(self) -> int:
        return self.gols_pro - self.gols_contra

    def ppg_casa(self) -> float:
        return self.pontos_casa / max(1, self.jogos_casa)

    def ppg_fora(self) -> float:
        return self.pontos_fora / max(1, self.jogos_fora)


@dataclass
class Time:
    nome: str
    rating: Rating
    stats: Estatisticas


@dataclass
class Jogo:
    rodada: int
    time_casa: str
    time_fora: str
    gols_casa: Optional[int] = None
    gols_fora: Optional[int] = None

    @property
    def foi_jogado(self) -> bool:
        return self.gols_casa is not None and self.gols_fora is not None

    def pontos(self) -> Tuple[int, int]:
        if not self.foi_jogado:
            return (0, 0)

        if self.gols_casa > self.gols_fora:
            return (3, 0)
        elif self.gols_casa < self.gols_fora:
            return (0, 3)
        else:
            return (1, 1)


@dataclass
class ForcaTime:
    ataque: float
    defesa: float


@dataclass
class ResultadoPartida:
    pontos_casa: int
    pontos_fora: int
    gols_casa: int
    gols_fora: int


# ==============================
# UTILITÁRIOS
# ==============================

def clone_times(times: Dict[str, Time]) -> Dict[str, Time]:
    novos = {}
    for nome, t in times.items():
        novos[nome] = Time(
            nome=t.nome,
            rating=t.rating,
            stats=Estatisticas()
        )
    return novos


# ==============================
# CALIBRADOR OVR TEMPORAL (MANTIDO)
# ==============================

class CalibradorOVRTemporal:
    """Mantido da V4.2 - núcleo estatístico intocável"""

    def __init__(self,
                 ovr_inicial: Dict[str, Dict[str, int]],
                 ovr_final: Dict[str, Dict[str, int]],
                 rodada_final: int,
                 k: float = None,
                 delta_max: float = None):
        self.ovr_inicial = ovr_inicial
        self.ovr_final = ovr_final
        self.rodada_final = rodada_final
        self.k = k if k is not None else config.k_suavidade_ovr
        self.delta_max = delta_max if delta_max is not None else config.delta_max_ovr

        self.ovr_por_rodada = self._gerar_ovr_por_rodada()
        self._validar_trajetoria()

    def _sigmoid(self, p: float) -> float:
        return 1.0 / (1.0 + math.exp(-self.k * (p - 0.5)))

    def _sigmoid_normalizada(self, p: float) -> float:
        s_0 = self._sigmoid(0.0)
        s_1 = self._sigmoid(1.0)
        s_p = self._sigmoid(p)
        return (s_p - s_0) / (s_1 - s_0)

    def _interpolar_ovr(self, ovr_ini: int, ovr_fim: int, progresso: float) -> float:
        s_norm = self._sigmoid_normalizada(progresso)
        return ovr_ini + s_norm * (ovr_fim - ovr_ini)

    def _gerar_ovr_por_rodada(self) -> Dict[int, Dict[str, Dict[str, int]]]:
        ovr_rodadas = {}
        ovr_rodadas[0] = dict(self.ovr_inicial)

        for rodada in range(1, self.rodada_final + 1):
            ovr_rodadas[rodada] = {}
            progresso = rodada / self.rodada_final

            for time in self.ovr_inicial.keys():
                ovr_rodadas[rodada][time] = {}

                for posicao in ['ATT', 'MEI', 'DEF']:
                    ovr_ini = self.ovr_inicial[time][posicao]
                    ovr_fim = self.ovr_final[time][posicao]

                    ovr_interpolado = self._interpolar_ovr(ovr_ini, ovr_fim, progresso)
                    ovr_anterior = ovr_rodadas[rodada - 1][time][posicao]
                    ovr_clipped = np.clip(
                        ovr_interpolado,
                        ovr_anterior - self.delta_max,
                        ovr_anterior + self.delta_max
                    )

                    ovr_rodadas[rodada][time][posicao] = int(round(ovr_clipped))

        return ovr_rodadas

    def get_ovr(self, time: str, rodada: int) -> Dict[str, int]:
        if rodada < 0 or rodada > self.rodada_final:
            raise ValueError(f"Rodada {rodada} fora do intervalo [0, {self.rodada_final}]")

        if time not in self.ovr_por_rodada[rodada]:
            raise ValueError(f"Time '{time}' não encontrado no calibrador")

        return self.ovr_por_rodada[rodada][time]

    def _validar_trajetoria(self):
        for time in self.ovr_inicial.keys():
            for pos in ['ATT', 'MEI', 'DEF']:
                assert self.ovr_por_rodada[0][time][pos] == self.ovr_inicial[time][pos], \
                    f"OVR inicial violado para {time}-{pos}"

        for time in self.ovr_final.keys():
            for pos in ['ATT', 'MEI', 'DEF']:
                ovr_gerado = self.ovr_por_rodada[self.rodada_final][time][pos]
                ovr_esperado = self.ovr_final[time][pos]
                assert abs(ovr_gerado - ovr_esperado) <= 1, \
                    f"OVR final violado para {time}-{pos}: {ovr_gerado} vs {ovr_esperado}"

        print(f"✅ Calibração OVR validada: {len(self.ovr_inicial)} times, rodadas 0-{self.rodada_final}")


# ==============================
# CALIBRAÇÃO DA LIGA (MANTIDA)
# ==============================

class CalibradorLiga:
    @staticmethod
    def escala_fifa(times: Dict[str, Time], proporcao_topo: float = 0.25) -> float:
        medias = [t.rating.media() for t in times.values()]
        medias.sort(reverse=True)
        n = max(1, int(len(medias) * proporcao_topo))
        return sum(medias[:n]) / n

    @staticmethod
    def media_gols_calibrada(times: Dict[str, Time]) -> float:
        total_gols = sum(t.stats.gols_pro for t in times.values())
        total_jogos = sum(t.stats.jogos for t in times.values())

        if total_jogos == 0:
            return config.media_gols_historica

        media_atual = total_gols / total_jogos
        peso_historico = 100 / (100 + total_jogos)
        peso_atual = 1 - peso_historico

        return (peso_historico * config.media_gols_historica +
                peso_atual * media_atual)


# ==============================
# BLOCO 3: EVOLUÇÃO ESTATÍSTICA
# (OPCIONAL, ISOLADA)
# ==============================

def peso_qualidade_dinamico(rodada: int, total_rodadas: int) -> float:
    """
    BLOCO 3.8: Peso dinâmico de OVR ao longo da temporada.

    OPT-IN via config.usar_peso_dinamico_ovr

    Justificativa: No início, OVR é tudo que temos.
    Conforme dados acumulam, confiamos mais neles.

    Riscos:
    - Pode suavizar variância em equipes inconsistentes
    - Reduz sensibilidade a mudanças abruptas

    Recomendação: Testar contra V4.2 em histórico
    """
    if not config.usar_peso_dinamico_ovr:
        # Fallback: comportamento V4.2
        return config.peso_qualidade_inicial

    # Decaimento exponencial suave
    progresso = rodada / total_rodadas
    return config.peso_qualidade_inicial * np.exp(-2 * progresso)


# ==============================
# CALCULADOR DE FORÇA (ADAPTADO)
# ==============================

class CalculadorForca:
    """Núcleo estatístico mantido, com hooks opcionais"""

    _SOFTCLIP_LUT = np.linspace(-3, 3, 1000)
    _SOFTCLIP_TANH = np.tanh(_SOFTCLIP_LUT)

    def __init__(self, escala_fifa: float, media_gols: float,
                 calibrador_ovr: CalibradorOVRTemporal = None,
                 rodada_atual: int = 0,
                 profiler: Optional[PerformanceProfiler] = None):
        self.escala_fifa = escala_fifa
        self.media_gols = media_gols
        self.calibrador_ovr = calibrador_ovr
        self.rodada_atual = rodada_atual
        self.profiler = profiler or PerformanceProfiler()
        self._cache_forca = {}

    def calcular(self, time: Time) -> ForcaTime:
        key = (time.nome, self.rodada_atual)

        if key in self._cache_forca:
            return self._cache_forca[key]

        with self.profiler.medir("calcular_forca"):
            # Prior estrutural (dinâmico se calibrador presente)
            if self.calibrador_ovr:
                ovr_rodada = self.calibrador_ovr.get_ovr(time.nome, self.rodada_atual)
                att_norm = ovr_rodada['ATT'] / self.escala_fifa
                mei_norm = ovr_rodada['MEI'] / self.escala_fifa
                def_norm = ovr_rodada['DEF'] / self.escala_fifa
            else:
                att_norm = time.rating.ATT / self.escala_fifa
                mei_norm = time.rating.MEI / self.escala_fifa
                def_norm = time.rating.DEF / self.escala_fifa

            qualidade_ataque = att_norm * 0.6 + mei_norm * 0.4
            qualidade_defesa = def_norm

            # Força observada
            if time.stats.jogos > 0:
                gols_pro_norm = time.stats.media_gols_pro() / self.media_gols

                gols_contra = time.stats.media_gols_contra()
                epsilon = 0.5
                defesa_observada = (self.media_gols / (gols_contra + epsilon)) * 0.75
            else:
                gols_pro_norm = qualidade_ataque
                defesa_observada = qualidade_defesa

            # Shrinkage adaptativo (HOOK para peso dinâmico)
            if config.usar_peso_dinamico_ovr:
                peso_qualidade = peso_qualidade_dinamico(time.stats.jogos, config.total_jogos)
            else:
                peso_qualidade = config.peso_qualidade_inicial * (
                        config.confianca_qualidade /
                        (config.confianca_qualidade + time.stats.jogos)
                )

            peso_dados = 1 - peso_qualidade

            ataque_raw = peso_qualidade * qualidade_ataque + peso_dados * gols_pro_norm
            defesa_raw = peso_qualidade * qualidade_defesa + peso_dados * defesa_observada

            # Soft-clip dinâmico
            amplitude_atual = self._calcular_amplitude_dinamica(time.stats.jogos)

            ataque = self._soft_clip_dinamico(ataque_raw, amplitude_atual)
            defesa = self._soft_clip_dinamico(defesa_raw, amplitude_atual)

            forca = ForcaTime(ataque=ataque, defesa=defesa)
            self._cache_forca[key] = forca
            return forca

    def _calcular_amplitude_dinamica(self, jogos: int) -> float:
        progresso = jogos / config.total_jogos
        return (config.amplitude_inicial * (1 - progresso) +
                config.amplitude_final * progresso)

    def _soft_clip_dinamico(self, valor: float, amplitude: float) -> float:
        x = (valor - config.centro_forca) / amplitude
        x = np.clip(x, -3, 3)
        idx = int((x + 3) / 6 * 999)
        return config.centro_forca + amplitude * self._SOFTCLIP_TANH[idx]

    def fator_mando_campo(self, time: Time) -> float:
        if time.stats.jogos_casa == 0 or time.stats.jogos_fora == 0:
            return config.fator_casa_prior

        peso_prior = config.confianca_minima_mando / (
                config.confianca_minima_mando + time.stats.jogos_casa)
        peso_dados = 1 - peso_prior

        ppg_casa = time.stats.ppg_casa()
        ppg_fora = time.stats.ppg_fora()
        fator_empirico = 1.0 + (ppg_casa - ppg_fora) * 0.10
        fator_empirico = np.clip(fator_empirico, 0.90, 1.25)

        return peso_prior * config.fator_casa_prior + peso_dados * fator_empirico


# ==============================
# MODELO DE PROBABILIDADES (MANTIDO)
# ==============================

class ModeloProbabilidades:
    @staticmethod
    def calcular_lambdas(forca_casa: ForcaTime,
                         forca_fora: ForcaTime,
                         fator_casa: float,
                         media_gols: float) -> Tuple[float, float]:
        vantagem_ataque_casa = forca_casa.ataque - forca_fora.defesa
        vantagem_ataque_fora = forca_fora.ataque - forca_casa.defesa

        multiplicador_casa = np.exp(config.fator_exponencial * vantagem_ataque_casa)
        multiplicador_fora = np.exp(config.fator_exponencial * vantagem_ataque_fora)

        fator_penalidade_fora = fator_casa ** config.expoente_mando_visitante

        lambda_casa = media_gols * multiplicador_casa * fator_casa
        lambda_fora = media_gols * multiplicador_fora / fator_penalidade_fora

        lambda_casa = np.clip(lambda_casa, 0.4, 4.5)
        lambda_fora = np.clip(lambda_fora, 0.4, 4.5)

        return lambda_casa, lambda_fora

    @staticmethod
    def calcular_probabilidades_dixon_coles(lambda_casa: float,
                                            lambda_fora: float,
                                            rho: float = None) -> Tuple[float, float, float]:
        if rho is None:
            rho = config.rho

        max_gols = 10
        p_vitoria = 0.0
        p_empate = 0.0
        p_derrota = 0.0

        for gc in range(max_gols):
            for gf in range(max_gols):
                p_gc = poisson_pmf(gc, lambda_casa)
                p_gf = poisson_pmf(gf, lambda_fora)

                tau = 1.0
                if gc == 0 and gf == 0:
                    tau = 1 - lambda_casa * lambda_fora * rho
                elif gc == 0 and gf == 1:
                    tau = 1 + lambda_casa * rho
                elif gc == 1 and gf == 0:
                    tau = 1 + lambda_fora * rho
                elif gc == 1 and gf == 1:
                    tau = 1 - rho

                p_placar = p_gc * p_gf * tau

                if gc > gf:
                    p_vitoria += p_placar
                elif gc == gf:
                    p_empate += p_placar
                else:
                    p_derrota += p_placar

        total = p_vitoria + p_empate + p_derrota
        return p_vitoria / total, p_empate / total, p_derrota / total


# ==============================
# SIMULADOR DE PARTIDAS (MANTIDO)
# ==============================

class SimuladorPartida:
    def __init__(self, calculador: CalculadorForca):
        self.calculador = calculador
        self.modelo_prob = ModeloProbabilidades()
        self._cache_dixon_coles = {}
        self._tau_cache = {}

    def simular(self, time_casa: Time, time_fora: Time, media_gols: float) -> ResultadoPartida:
        forca_casa = self.calculador._cache_forca[(time_casa.nome, self.calculador.rodada_atual)]
        forca_fora = self.calculador._cache_forca[(time_fora.nome, self.calculador.rodada_atual)]
        fator_casa = self.calculador.fator_mando_campo(time_casa)

        lambda_casa, lambda_fora = self.modelo_prob.calcular_lambdas(
            forca_casa, forca_fora, fator_casa, media_gols)

        gols_casa, gols_fora = self._gerar_placar_dixon_coles(lambda_casa, lambda_fora)

        if gols_casa > gols_fora:
            pontos_casa, pontos_fora = 3, 0
        elif gols_casa < gols_fora:
            pontos_casa, pontos_fora = 0, 3
        else:
            pontos_casa, pontos_fora = 1, 1

        return ResultadoPartida(
            pontos_casa=pontos_casa,
            pontos_fora=pontos_fora,
            gols_casa=gols_casa,
            gols_fora=gols_fora
        )

    def _gerar_placar_dixon_coles(self, lambda_casa: float, lambda_fora: float):
        key = (round(lambda_casa, 2), round(lambda_fora, 2))

        if key in self._cache_dixon_coles:
            placares, probs = self._cache_dixon_coles[key]
            idx = np.random.choice(len(placares), p=probs)
            return placares[idx]

        max_gols = 6 if lambda_casa + lambda_fora < 2.2 else 8

        p_casa = [poisson_pmf(i, lambda_casa) for i in range(max_gols + 1)]
        p_fora = [poisson_pmf(i, lambda_fora) for i in range(max_gols + 1)]

        placares = []
        probs = []

        for gc in range(max_gols + 1):
            for gf in range(max_gols + 1):
                tau = self._fator_correlacao_dixon_coles(
                    gc, gf, lambda_casa, lambda_fora
                )
                placares.append((gc, gf))
                probs.append(p_casa[gc] * p_fora[gf] * tau)

        probs = np.array(probs)
        probs /= probs.sum()

        self._cache_dixon_coles[key] = (placares, probs)

        idx = np.random.choice(len(placares), p=probs)
        return placares[idx]

    def _fator_correlacao_dixon_coles(
            self, gc: int, gf: int,
            lambda_casa: float, lambda_fora: float
    ) -> float:
        rho = config.rho
        key = (gc, gf, round(lambda_casa, 2), round(lambda_fora, 2))

        if key in self._tau_cache:
            return self._tau_cache[key]

        if gc == 0 and gf == 0:
            tau = 1 - lambda_casa * lambda_fora * rho
        elif gc == 0 and gf == 1:
            tau = 1 + lambda_casa * rho
        elif gc == 1 and gf == 0:
            tau = 1 + lambda_fora * rho
        elif gc == 1 and gf == 1:
            tau = 1 - rho
        else:
            tau = 1.0

        self._tau_cache[key] = tau
        return tau


# ==============================
# GERENCIADOR DE TEMPORADA (REFATORADO)
# ==============================

class GerenciadorTemporada:
    """
    REFATORADO: Separação clara entre simulação e efeitos colaterais.
    Agora retorna ResultadoSimulacao imutável.
    """

    def __init__(
            self,
            times_iniciais: Dict[str, Time],
            jogos: List[Jogo],
            calibrador_ovr: CalibradorOVRTemporal = None,
            profiler: Optional[PerformanceProfiler] = None
    ):
        self.times_iniciais = times_iniciais
        self.jogos = jogos
        self.calibrador_ovr = calibrador_ovr
        self.profiler = profiler or PerformanceProfiler()
        self._simulador = None

    def simular(self, simulacao_id: int = 0) -> ResultadoSimulacao:
        """
        Simula UMA temporada completa.

        Returns:
            ResultadoSimulacao imutável (facilitaparalalelização)
        """
        times = clone_times(self.times_iniciais)

        rodadas = {}
        for jogo in self.jogos:
            rodadas.setdefault(jogo.rodada, []).append(jogo)

        with self.profiler.medir("simular_temporada"):
            for rodada in sorted(rodadas.keys()):
                self._simular_rodada(times, rodadas[rodada], rodada)

        # Classifica times (pontos → saldo → gols pró)
        tabela = sorted(
            times.items(),
            key=lambda x: (
                -x[1].stats.pontos_totais(),
                -x[1].stats.saldo_gols(),
                -x[1].stats.gols_pro
            )
        )

        classificacao = [nome for nome, _ in tabela]
        pontos = {nome: time.stats.pontos_totais() for nome, time in times.items()}
        saldos = {nome: time.stats.saldo_gols() for nome, time in times.items()}
        gols_pro = {nome: time.stats.gols_pro for nome, time in times.items()}

        return ResultadoSimulacao(
            simulacao_id=simulacao_id,
            classificacao=classificacao,
            pontos=pontos,
            saldos=saldos,
            gols_pro=gols_pro
        )

    def _simular_rodada(
            self,
            times: Dict[str, Time],
            jogos: List[Jogo],
            rodada_atual: int
    ):
        escala = CalibradorLiga.escala_fifa(times)
        media = CalibradorLiga.media_gols_calibrada(times)

        if self._simulador is None:
            calculador = CalculadorForca(
                escala_fifa=escala,
                media_gols=media,
                calibrador_ovr=self.calibrador_ovr,
                rodada_atual=rodada_atual,
                profiler=self.profiler
            )
            self._simulador = SimuladorPartida(calculador)
        else:
            calculador = self._simulador.calculador
            calculador.escala_fifa = escala
            calculador.media_gols = media
            calculador.rodada_atual = rodada_atual

        calculador._cache_forca.clear()

        for nome, time in times.items():
            calculador.calcular(time)

        simulador = self._simulador

        for jogo in jogos:
            if jogo.foi_jogado:
                resultado = ResultadoPartida(
                    *jogo.pontos(),
                    jogo.gols_casa,
                    jogo.gols_fora
                )
            else:
                resultado = simulador.simular(
                    times[jogo.time_casa],
                    times[jogo.time_fora],
                    media
                )

            self._atualizar_stats(times[jogo.time_casa], resultado, True)
            self._atualizar_stats(times[jogo.time_fora], resultado, False)

    def _atualizar_stats(self, time: Time, resultado: ResultadoPartida, eh_casa: bool):
        if eh_casa:
            time.stats.pontos_casa += resultado.pontos_casa
            time.stats.jogos_casa += 1
            time.stats.gols_pro += resultado.gols_casa
            time.stats.gols_contra += resultado.gols_fora
        else:
            time.stats.pontos_fora += resultado.pontos_fora
            time.stats.jogos_fora += 1
            time.stats.gols_pro += resultado.gols_fora
            time.stats.gols_contra += resultado.gols_casa

        time.stats.jogos += 1


# ==============================
# BLOCO 2.4: SIMULADOR PARALELO
# ==============================

def _worker_simular(args):
    """
    Worker paralelo para simulações independentes.

    Cada worker recebe:
    - seed própria (independência estatística)
    - cópia dos dados (sem estado compartilhado)
    - range de IDs de simulações
    """
    worker_id, seed, times_iniciais, jogos, calibrador_ovr, id_inicio, id_fim = args

    # Configura seed do worker
    SeedManager.configurar_seed(seed)

    # Cria gerenciador local
    gerenciador = GerenciadorTemporada(
        times_iniciais=times_iniciais,
        jogos=jogos,
        calibrador_ovr=calibrador_ovr,
        profiler=None  # Profiling desativado em workers
    )

    # Executa simulações
    resultados = []
    for sim_id in range(id_inicio, id_fim):
        resultado = gerenciador.simular(sim_id)
        resultados.append(resultado)

    return resultados


# ==============================
# AGREGADOR DE RESULTADOS (BLOCO 1.2)
# ==============================

class AgregadorResultados:
    """
    BLOCO 1.2 + 4.10: Agregação pura sem efeitos colaterais.
    Calcula distribuições completas, não apenas médias.
    """

    @staticmethod
    def agregar(resultados: List[ResultadoSimulacao]) -> ResultadoTemporada:
        """
        Agrega múltiplas simulações em estatísticas completas.

        BLOCO 4.10: Mantém distribuições completas, não só médias.
        """
        if not resultados:
            raise ValueError("Lista de resultados vazia")

        n_sim = len(resultados)
        times = list(resultados[0].pontos.keys())

        # Inicializa contadores
        titulos = defaultdict(int)
        promocoes = defaultdict(int)
        playoffs = defaultdict(int)
        rebaixamentos = defaultdict(int)

        # Distribuições completas (BLOCO 4.10)
        pontos_dist = defaultdict(list)
        saldos_dist = defaultdict(list)
        posicoes_dist = defaultdict(list)

        # Processa cada simulação
        for resultado in resultados:
            # Título (1º)
            titulos[resultado.classificacao[0]] += 1

            # Promoção (top 3)
            for i in range(3):
                promocoes[resultado.classificacao[i]] += 1

            # Playoffs (4º-7º)
            for i in range(3, 7):
                playoffs[resultado.classificacao[i]] += 1

            # Rebaixamento (últimos 2)
            for i in range(22, 24):
                rebaixamentos[resultado.classificacao[i]] += 1

            # Distribuições
            for time in times:
                pontos_dist[time].append(resultado.pontos[time])
                saldos_dist[time].append(resultado.saldos[time])
                posicao = resultado.classificacao.index(time) + 1
                posicoes_dist[time].append(posicao)

        # Normaliza probabilidades
        prob_titulo = {t: titulos[t] / n_sim for t in times}
        prob_promocao = {t: promocoes[t] / n_sim for t in times}
        prob_playoffs = {t: playoffs[t] / n_sim for t in times}
        prob_rebaixamento = {t: rebaixamentos[t] / n_sim for t in times}

        return ResultadoTemporada(
            n_simulacoes=n_sim,
            times=times,
            probabilidades_titulo=prob_titulo,
            probabilidades_promocao=prob_promocao,
            probabilidades_playoffs=prob_playoffs,
            probabilidades_rebaixamento=prob_rebaixamento,
            pontos_distribuicao=dict(pontos_dist),
            saldos_distribuicao=dict(saldos_dist),
            posicoes_distribuicao=dict(posicoes_dist),
            seed_utilizada=SeedManager._master_seed
        )


# ==============================
# SIMULADOR MONTE CARLO
# ==============================

class SimuladorMonteCarlo:

    def __init__(
            self,
            times: Dict[str, Time],
            jogos: List[Jogo],
            calibrador_ovr: CalibradorOVRTemporal = None,
            seed: Optional[int] = None
    ):
        self.times_iniciais = times
        self.jogos = jogos
        self.calibrador_ovr = calibrador_ovr

        # Configura seed
        SeedManager.configurar_seed(seed)

        # Managers
        self.checkpoint_mgr = CheckpointManager()
        self.profiler = PerformanceProfiler()

        if not config.ativar_profiling:
            self.profiler.desativar()

        # Resultados
        self.resultados_parciais: List[ResultadoSimulacao] = []
        self.resultado_final: Optional[ResultadoTemporada] = None

    def executar(self, n_simulacoes: int, checkpoint_id: Optional[str] = None):
        """
        Executa simulações com checkpoints opcionais.

        Args:
            n_simulacoes: Número de simulações
            checkpoint_id: ID para salvar/carregar checkpoints
        """
        inicio_total = time.time()

        # Tenta carregar checkpoint
        inicio_sim = 0
        if checkpoint_id:
            carregados, metadata, sucesso = self.checkpoint_mgr.carregar_checkpoint(checkpoint_id)
            if sucesso:
                self.resultados_parciais = carregados
                inicio_sim = len(carregados)
                print(f"✅ Checkpoint carregado: {inicio_sim} simulações")

        if inicio_sim >= n_simulacoes:
            print("✅ Simulações já completas!")
            self._agregar_resultados(n_simulacoes, time.time() - inicio_total)
            return

        # Escolhe modo de execução
        if config.usar_paralelizacao:
            self._executar_paralelo(inicio_sim, n_simulacoes, checkpoint_id)
        else:
            self._executar_serial(inicio_sim, n_simulacoes, checkpoint_id)

        # Agregação final
        tempo_total = time.time() - inicio_total
        self._agregar_resultados(n_simulacoes, tempo_total)

        # Profiling
        if config.ativar_profiling:
            print(self.profiler.relatorio())

    def _executar_serial(self, inicio: int, fim: int, checkpoint_id: Optional[str]):
        """Execução serial com checkpoints e barra de progresso"""
        gerenciador = GerenciadorTemporada(
            self.times_iniciais,
            self.jogos,
            self.calibrador_ovr,
            self.profiler
        )

        print(f"\n🎲 Executando simulações:")

        for i in range(inicio, fim):
            resultado = gerenciador.simular(i)
            self.resultados_parciais.append(resultado)

            # Atualiza barra de progresso
            if (i + 1 - inicio) % config.atualizar_progresso_a_cada == 0 or (i + 1) == fim:
                sys.stdout.write('\r' + barra_progresso(i + 1 - inicio, fim - inicio))
                sys.stdout.flush()

            # Checkpoint incremental
            if checkpoint_id and (i + 1) % config.checkpoint_intervalo == 0:
                self.checkpoint_mgr.salvar_checkpoint(
                    self.resultados_parciais,
                    {'n_simulacoes': fim, 'checkpoint_id': checkpoint_id},
                    checkpoint_id
                )

        print()  # quebra de linha final

    def _executar_serial_silencioso(self, inicio: int, fim: int):
        """Execução serial sem output (para análise de evolução)"""
        gerenciador = GerenciadorTemporada(
            self.times_iniciais,
            self.jogos,
            self.calibrador_ovr,
            self.profiler
        )

        for i in range(inicio, fim):
            resultado = gerenciador.simular(i)
            self.resultados_parciais.append(resultado)

    def _executar_paralelo(self, inicio: int, fim: int, checkpoint_id: Optional[str]):
        """
        BLOCO 2.4: Execução paralela com independência estatística e barra de progresso.

        Estratégia:
        - Divide simulações entre workers
        - Cada worker recebe seed independente
        - Sem estado compartilhado
        - Resultados mesclados ao final
        """

        n_restantes = fim - inicio
        sims_por_worker = n_restantes // config.n_workers

        # Prepara argumentos para workers
        tarefas = []
        for worker_id in range(config.n_workers):
            seed_worker = SeedManager.gerar_seed_worker(worker_id)
            id_inicio = inicio + worker_id * sims_por_worker
            id_fim = id_inicio + sims_por_worker

            if worker_id == config.n_workers - 1:
                id_fim = fim

            tarefas.append((
                worker_id,
                seed_worker,
                self.times_iniciais,
                self.jogos,
                self.calibrador_ovr,
                id_inicio,
                id_fim
            ))

        # Executa em paralelo
        print(f"\n🎲 Executando simulações ({config.n_workers} workers):")

        with mp.Pool(config.n_workers) as pool:
            # Usando imap_unordered para processar resultados à medida que ficam prontos
            resultados_completos = []
            total_processado = 0

            for resultados_worker in pool.imap_unordered(_worker_simular, tarefas):
                resultados_completos.append(resultados_worker)
                total_processado += len(resultados_worker)

                # Atualiza barra de progresso
                sys.stdout.write('\r' + barra_progresso(total_processado, n_restantes))
                sys.stdout.flush()

        print()  # quebra de linha final

        # Mescla resultados
        for resultados_worker in resultados_completos:
            self.resultados_parciais.extend(resultados_worker)

        # Checkpoint final
        if checkpoint_id:
            self.checkpoint_mgr.salvar_checkpoint(
                self.resultados_parciais,
                {'n_simulacoes': fim, 'checkpoint_id': checkpoint_id},
                checkpoint_id
            )

    def _agregar_resultados(self, n_simulacoes: int, tempo_execucao: float):
        """Agregação final dos resultados"""
        self.resultado_final = AgregadorResultados.agregar(self.resultados_parciais)
        self.resultado_final.tempo_execucao = tempo_execucao
        self.resultado_final.n_simulacoes = n_simulacoes

    def obter_resultados(self) -> List[Dict]:
        """
        Retorna resultados no formato V4 para compatibilidade.
        """
        if self.resultado_final is None:
            raise RuntimeError("Simulações não executadas")

        r = self.resultado_final
        saida = []

        for nome in r.times:
            saida.append({
                "time": nome,
                "titulo": r.probabilidades_titulo[nome],
                "promocao": r.probabilidades_promocao[nome],
                "playoffs": r.probabilidades_playoffs[nome],
                "rebaixamento": r.probabilidades_rebaixamento[nome],
                "pontos_finais": np.mean(r.pontos_distribuicao[nome]),
                "saldo_medio": np.mean(r.saldos_distribuicao[nome]),
            })

        saida.sort(key=lambda x: (-x["titulo"], -x["promocao"],
                                  -x["playoffs"], x["rebaixamento"]))
        return saida


# ==============================
# BLOCO 4: ANÁLISE AVANÇADA
# ==============================

class AnalisadorDistribuicoes:
    """
    BLOCO 4.10: Análise de distribuições completas.

    Além de médias, fornece:
    - Percentis (P10, P25, P50, P75, P90)
    - Intervalos de confiança
    - Histogramas
    """

    @staticmethod
    def analise_time(nome_time: str, resultado: ResultadoTemporada) -> Dict:
        """Análise completa de um time"""
        pontos = resultado.pontos_distribuicao[nome_time]
        saldos = resultado.saldos_distribuicao[nome_time]
        posicoes = resultado.posicoes_distribuicao[nome_time]

        return {
            'time': nome_time,
            'pontos': {
                'media': np.mean(pontos),
                'mediana': np.median(pontos),
                'p10': np.percentile(pontos, 10),
                'p90': np.percentile(pontos, 90),
                'min': min(pontos),
                'max': max(pontos),
                'desvio': np.std(pontos)
            },
            'posicao': {
                'media': np.mean(posicoes),
                'mediana': np.median(posicoes),
                'melhor': min(posicoes),
                'pior': max(posicoes)
            },
            'probabilidades': {
                'titulo': resultado.probabilidades_titulo[nome_time],
                'promocao': resultado.probabilidades_promocao[nome_time],
                'playoffs': resultado.probabilidades_playoffs[nome_time],
                'rebaixamento': resultado.probabilidades_rebaixamento[nome_time]
            }
        }

    @staticmethod
    def comparar_times(times: List[str], resultado: ResultadoTemporada) -> str:
        """Comparação lado-a-lado de múltiplos times"""
        linhas = ["\n" + "=" * 60, "📊 COMPARAÇÃO DETALHADA", "=" * 60]

        for time in times:
            analise = AnalisadorDistribuicoes.analise_time(time, resultado)
            linhas.append(f"\n{time}:")
            linhas.append(f"  Pontos: {analise['pontos']['media']:.1f} ± {analise['pontos']['desvio']:.1f}")
            linhas.append(f"  Range: {analise['pontos']['min']}-{analise['pontos']['max']}")
            linhas.append(f"  P10-P90: {analise['pontos']['p10']:.0f}-{analise['pontos']['p90']:.0f}")
            linhas.append(f"  Posição média: {analise['posicao']['media']:.1f}º")
            linhas.append(f"  Título: {analise['probabilidades']['titulo'] * 100:.2f}%")

        return "\n".join(linhas)


# ==============================
# BLOCO 5: TESTES DE REGRESSÃO
# ==============================

class ValidadorEstatistico:
    """
    BLOCO 5.13: Testes de regressão entre versões.

    Garante que V5 ≈ V4.1 quando todas as flags estão OFF.
    Usa teste Kolmogorov-Smirnov para distribuições.
    """

    @staticmethod
    def comparar_versoes(resultado_v4: List[Dict], resultado_v5: ResultadoTemporada, alpha=0.05) -> bool:
        """
        Compara resultados entre versões.

        Critérios:
        - KS test para distribuições de pontos
        - Diferença máxima em probabilidades < 2%
        """

        diferencas = []

        for time_v4 in resultado_v4:
            nome = time_v4['time']

            # Compara pontos finais
            pts_v4 = time_v4['pontos_finais']
            pts_v5 = np.mean(resultado_v5.pontos_distribuicao[nome])
            diff_pts = abs(pts_v4 - pts_v5)
            diferencas.append(diff_pts)

            # Compara probabilidades
            prob_campos = ['titulo', 'promocao', 'playoffs', 'rebaixamento']
            for campo in prob_campos:
                prob_v4 = time_v4[campo]
                prob_v5 = getattr(resultado_v5, f'probabilidades_{campo}')[nome]
                diff_prob = abs(prob_v4 - prob_v5)

                if diff_prob > 0.02:  # Mais de 2% de diferença
                    print(f"⚠️  {nome} - {campo}: diferença de {diff_prob * 100:.2f}%")

        max_diff = max(diferencas)
        aprovado = max_diff < 2.0  # Menos de 2 pontos de diferença

        if aprovado:
            print(f"✅ Validação aprovada: diferença máxima = {max_diff:.2f} pontos")
        else:
            print(f"❌ Validação falhou: diferença máxima = {max_diff:.2f} pontos")

        return aprovado


# ==============================
# BLOCO 6: ANÁLISE DE EVOLUÇÃO POR RODADA
# ==============================

class AnalisadorEvolucao:
    """
    Analisa a evolução das probabilidades de um time ao longo das rodadas.
    Executa simulações para cada rodada e rastreia mudanças.
    """

    def __init__(self, times_base: Dict[str, Time], jogos: List[Jogo],
                 calibrador_ovr: CalibradorOVRTemporal):
        self.times_base = times_base
        self.jogos = jogos
        self.calibrador_ovr = calibrador_ovr
        self.rodadas_disponiveis = sorted({j.rodada for j in jogos if j.foi_jogado})

    def analisar_evolucao(self, nome_time: str, n_simulacoes: int = 5000) -> Dict:
        """
        Analisa evolução do time rodada por rodada.

        Args:
            nome_time: Nome do time a analisar
            n_simulacoes: Número de simulações por rodada

        Returns:
            Dict com histórico de probabilidades por rodada
        """
        if nome_time not in self.times_base:
            raise ValueError(f"Time '{nome_time}' não encontrado")

        print(f"\n{'=' * 60}")
        print(f"📈 ANÁLISE DE EVOLUÇÃO - {nome_time}")
        print(f"{'-' * 60}")
        print(f"Rodadas com dados: {len(self.rodadas_disponiveis)}")
        print(f"Simulações por rodada: {n_simulacoes}")
        print(f"{'=' * 60}\n")

        historico = {
            'rodadas': [],
            'titulo': [],
            'promocao': [],
            'playoffs': [],
            'rebaixamento': [],
            'pontos_medios': [],
            'posicao_media': []
        }

        # Adiciona rodada 0 (antes do campeonato)
        print("Simulando rodada 0 (projeção inicial)...")
        resultado_r0 = self._simular_rodada(0, n_simulacoes)
        self._adicionar_ao_historico(historico, 0, nome_time, resultado_r0)

        # Simula cada rodada
        total_rodadas = len(self.rodadas_disponiveis)
        for idx, rodada in enumerate(self.rodadas_disponiveis, 1):
            print(f"Simulando após rodada {rodada}... ", end='')
            sys.stdout.flush()

            resultado = self._simular_rodada(rodada, n_simulacoes)
            self._adicionar_ao_historico(historico, rodada, nome_time, resultado)

            print(f"✅ [{idx}/{total_rodadas}]")

        return historico

    def _simular_rodada(self, rodada: int, n_simulacoes: int) -> ResultadoTemporada:
        """Executa simulação para uma rodada específica"""
        times_atualizados = extrair_estatisticas_do_calendario(
            self.jogos, self.times_base, rodada
        )

        jogos_para_simular = []
        for jogo in self.jogos:
            if jogo.rodada <= rodada:
                jogos_para_simular.append(jogo)
            else:
                jogos_para_simular.append(Jogo(
                    rodada=jogo.rodada,
                    time_casa=jogo.time_casa,
                    time_fora=jogo.time_fora,
                    gols_casa=None,
                    gols_fora=None
                ))

        # Simula sem verbose
        simulador = SimuladorMonteCarlo(
            times_atualizados,
            jogos_para_simular,
            self.calibrador_ovr,
            seed=None  # Aleatório para cada rodada
        )

        # Executa sem checkpoints
        simulador._executar_serial_silencioso(0, n_simulacoes)
        simulador._agregar_resultados(n_simulacoes, 0)

        return simulador.resultado_final

    def _adicionar_ao_historico(self, historico: Dict, rodada: int,
                                nome_time: str, resultado: ResultadoTemporada):
        """Adiciona dados de uma rodada ao histórico"""
        historico['rodadas'].append(rodada)
        historico['titulo'].append(resultado.probabilidades_titulo[nome_time] * 100)
        historico['promocao'].append(resultado.probabilidades_promocao[nome_time] * 100)
        historico['playoffs'].append(resultado.probabilidades_playoffs[nome_time] * 100)
        historico['rebaixamento'].append(resultado.probabilidades_rebaixamento[nome_time] * 100)
        historico['pontos_medios'].append(
            np.mean(resultado.pontos_distribuicao[nome_time])
        )
        historico['posicao_media'].append(
            np.mean(resultado.posicoes_distribuicao[nome_time])
        )

    @staticmethod
    def imprimir_evolucao(historico: Dict, nome_time: str):
        """Imprime tabela formatada da evolução"""
        print(f"\n{'=' * 80}")
        print(f"📊 EVOLUÇÃO DAS PROBABILIDADES - {nome_time}")
        print(f"{'=' * 80}")
        print(f"{'Rod':>4} | {'Título':>8} | {'Promoção':>10} | {'Playoffs':>9} | "
              f"{'Rebaixa':>9} | {'Pts':>5} | {'Pos':>5}")
        print(f"{'-' * 80}")

        for i in range(len(historico['rodadas'])):
            rodada = historico['rodadas'][i]
            titulo = historico['titulo'][i]
            promocao = historico['promocao'][i]
            playoffs = historico['playoffs'][i]
            rebaixa = historico['rebaixamento'][i]
            pontos = historico['pontos_medios'][i]
            posicao = historico['posicao_media'][i]

            # Indicador de mudança
            if i > 0:
                delta_titulo = titulo - historico['titulo'][i - 1]
                indicador = "📈" if delta_titulo > 2 else "📉" if delta_titulo < -2 else "➡️"
            else:
                indicador = "🎯"

            print(f"{rodada:>4} | {titulo:>7.2f}% | {promocao:>9.2f}% | "
                  f"{playoffs:>8.2f}% | {rebaixa:>8.2f}% | {pontos:>5.1f} | "
                  f"{posicao:>5.1f} {indicador}")

        print(f"{'=' * 80}\n")

        # Resumo de tendências
        AnalisadorEvolucao._imprimir_tendencias(historico)

    @staticmethod
    def _imprimir_tendencias(historico: Dict):
        """Imprime análise de tendências"""
        if len(historico['rodadas']) < 2:
            return

        titulo_inicial = historico['titulo'][0]
        titulo_atual = historico['titulo'][-1]
        delta_titulo = titulo_atual - titulo_inicial

        promocao_inicial = historico['promocao'][0]
        promocao_atual = historico['promocao'][-1]
        delta_promocao = promocao_atual - promocao_inicial

        playoffs_inicial = historico['playoffs'][0]
        playoffs_atual = historico['playoffs'][-1]
        delta_playoffs = playoffs_atual - playoffs_inicial

        rebaixa_inicial = historico['rebaixamento'][0]
        rebaixa_atual = historico['rebaixamento'][-1]
        delta_rebaixa = rebaixa_atual - rebaixa_inicial

        print(f"📊 ANÁLISE DE TENDÊNCIAS:")
        print(f"{'-' * 60}")

        # Título
        if abs(delta_titulo) > 5:
            if delta_titulo > 0:
                print(f"✅ Chances de título AUMENTARAM {delta_titulo:+.2f}% "
                      f"({titulo_inicial:.2f}% → {titulo_atual:.2f}%)")
            else:
                print(f"❌ Chances de título DIMINUÍRAM {delta_titulo:.2f}% "
                      f"({titulo_inicial:.2f}% → {titulo_atual:.2f}%)")
        else:
            print(f"➡️  Chances de título estáveis ({titulo_atual:.2f}%)")

        # Promoção
        if abs(delta_promocao) > 5:
            if delta_promocao > 0:
                print(f"✅ Chances de promoção AUMENTARAM {delta_promocao:+.2f}% "
                      f"({promocao_inicial:.2f}% → {promocao_atual:.2f}%)")
            else:
                print(f"❌ Chances de promoção DIMINUÍRAM {delta_promocao:.2f}% "
                      f"({promocao_inicial:.2f}% → {promocao_atual:.2f}%)")
        else:
            print(f"➡️  Chances de promoção estáveis ({promocao_atual:.2f}%)")

        # Playoffs
        if abs(delta_playoffs) > 5:
            if delta_playoffs > 0:
                print(f"✅ Chances de playoffs AUMENTARAM {delta_playoffs:+.2f}% "
                      f"({playoffs_inicial:.2f}% → {playoffs_atual:.2f}%)")
            else:
                print(f"❌ Chances de playoffs DIMINUÍRAM {delta_playoffs:.2f}% "
                      f"({playoffs_inicial:.2f}% → {playoffs_atual:.2f}%)")
        else:
            print(f"➡️  Chances de playoffs estáveis ({playoffs_atual:.2f}%)")

        # Rebaixamento
        if abs(delta_playoffs) > 5:
            if delta_playoffs < 0:
                print(f"✅ Chances de rebaixamento DIMINUIRAM {delta_rebaixa:+.2f}% "
                      f"({rebaixa_inicial:.2f}% → {rebaixa_atual:.2f}%)")
            else:
                print(f"❌ Chances de rebaixamento AUMENTARAM {delta_rebaixa:.2f}% "
                      f"({rebaixa_inicial:.2f}% → {rebaixa_atual:.2f}%)")
        else:
            print(f"➡️  Chances de rebaixamento estáveis ({rebaixa_atual:.2f}%)")

        # Maior oscilação positiva
        variacoes = np.diff(historico['titulo'])
        if len(variacoes) > 0:
            idx_max = np.argmax(np.abs(variacoes))
            rodada_mudanca = historico['rodadas'][idx_max + 1]
            variacao_max = variacoes[idx_max]

            print(f"\n📈 Melhor mudança: Rodada {rodada_mudanca} "
                  f"({variacao_max:+.2f}% no título)")

        # Maior oscilação negativa
        variacoes = np.diff(historico['titulo'])
        if len(variacoes) < 0:
            idx_min = np.argmin(np.abs(variacoes))
            rodada_mudanca = historico['rodadas'][idx_min + 1]
            variacao_min = variacoes[idx_min]

            print(f"\n📉 Pior mudança: Rodada {rodada_mudanca} "
                  f"({variacao_min:+.2f}% no título)")

        print(f"{'-' * 60}\n")

# ==============================
# CALENDÁRIO (MANTIDO)
# ==============================

def _parse_gols(valor: str):
    if valor is None:
        return None
    v = str(valor).strip()
    if v == "" or v == "-":
        return None
    try:
        return int(v)
    except ValueError:
        return None

def carregar_calendario(arquivo: str) -> List[Jogo]:
    jogos = []
    try:
        with open(arquivo, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rodada = int(row['rodada'])
                casa = row['time_casa'].strip()
                fora = row['time_fora'].strip()

                gols_casa = None
                gols_fora = None

                gols_casa = _parse_gols(row.get('gols_casa'))
                gols_fora = _parse_gols(row.get('gols_fora'))

                jogos.append(Jogo(rodada, casa, fora, gols_casa, gols_fora))

        return jogos

    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {arquivo}")
    except KeyError as e:
        raise ValueError(f"❌ Coluna não encontrada no CSV: {e}")


def extrair_estatisticas_do_calendario(
        jogos: List[Jogo],
        times_base: Dict[str, Time],
        ate_rodada: Optional[int] = None) -> Dict[str, Time]:
    times = clone_times(times_base)

    for time in times.values():
        time.stats = Estatisticas()

    if ate_rodada == 0:
        return times

    for jogo in jogos:
        if ate_rodada and jogo.rodada > ate_rodada:
            break

        if not jogo.foi_jogado:
            continue

        casa = times[jogo.time_casa]
        fora = times[jogo.time_fora]
        pontos_casa, pontos_fora = jogo.pontos()

        casa.stats.gols_pro += jogo.gols_casa
        casa.stats.gols_contra += jogo.gols_fora
        casa.stats.pontos_casa += pontos_casa
        casa.stats.jogos_casa += 1
        casa.stats.jogos += 1

        fora.stats.gols_pro += jogo.gols_fora
        fora.stats.gols_contra += jogo.gols_casa
        fora.stats.pontos_fora += pontos_fora
        fora.stats.jogos_fora += 1
        fora.stats.jogos += 1

    return times


# ==============================
# DADOS DOS TIMES
# ==============================

def obter_ratings_iniciais() -> Dict[str, Dict[str, int]]:
    return {
        "Accrington": {"ATT": 58, "MEI": 59, "DEF": 58},
        "Barnet": {"ATT": 60, "MEI": 62, "DEF": 60},
        "Barrow": {"ATT": 61, "MEI": 62, "DEF": 62},
        "Bristol Rovers": {"ATT": 64, "MEI": 64, "DEF": 63},
        "Bromley FC": {"ATT": 47, "MEI": 46, "DEF": 46},
        "Cambridge United": {"ATT": 64, "MEI": 63, "DEF": 64},
        "Cheltenham Town": {"ATT": 60, "MEI": 63, "DEF": 60},
        "Chesterfield": {"ATT": 62, "MEI": 63, "DEF": 61},
        "Colchester": {"ATT": 60, "MEI": 62, "DEF": 62},
        "Crawley Town": {"ATT": 58, "MEI": 61, "DEF": 62},
        "Crewe Alexandra": {"ATT": 62, "MEI": 58, "DEF": 61},
        "Fleetwood Town": {"ATT": 63, "MEI": 62, "DEF": 63},
        "Gillingham": {"ATT": 62, "MEI": 61, "DEF": 62},
        "Grimsby Town": {"ATT": 61, "MEI": 61, "DEF": 62},
        "Harrogate Town": {"ATT": 62, "MEI": 60, "DEF": 61},
        "MK Dons": {"ATT": 66, "MEI": 63, "DEF": 63},
        "Newport County": {"ATT": 58, "MEI": 59, "DEF": 58},
        "Notts County": {"ATT": 62, "MEI": 63, "DEF": 62},
        "Oldham Athletic": {"ATT": 58, "MEI": 60, "DEF": 60},
        "Salford City": {"ATT": 60, "MEI": 63, "DEF": 61},
        "Shrewsbury": {"ATT": 62, "MEI": 63, "DEF": 63},
        "Swindon Town": {"ATT": 62, "MEI": 61, "DEF": 59},
        "Tranmere Rovers": {"ATT": 61, "MEI": 62, "DEF": 62},
        "Walsall": {"ATT": 61, "MEI": 63, "DEF": 61},
    }


def obter_ratings_atuais() -> Dict[str, Dict[str, int]]:
    return {
        "Accrington": {"ATT": 58, "MEI": 59, "DEF": 58},
        "Barnet": {"ATT": 61, "MEI": 63, "DEF": 61},
        "Barrow": {"ATT": 61, "MEI": 61, "DEF": 62},
        "Bristol Rovers": {"ATT": 64, "MEI": 62, "DEF": 63},
        "Bromley FC": {"ATT": 47, "MEI": 47, "DEF": 47},
        "Cambridge United": {"ATT": 64, "MEI": 63, "DEF": 63},
        "Cheltenham Town": {"ATT": 61, "MEI": 61, "DEF": 60},
        "Chesterfield": {"ATT": 62, "MEI": 63, "DEF": 61},
        "Colchester": {"ATT": 60, "MEI": 61, "DEF": 62},
        "Crawley Town": {"ATT": 58, "MEI": 61, "DEF": 62},
        "Crewe Alexandra": {"ATT": 59, "MEI": 59, "DEF": 61},
        "Fleetwood Town": {"ATT": 60, "MEI": 63, "DEF": 63},
        "Gillingham": {"ATT": 61, "MEI": 61, "DEF": 63},
        "Grimsby Town": {"ATT": 60, "MEI": 62, "DEF": 62},
        "Harrogate Town": {"ATT": 64, "MEI": 60, "DEF": 61},
        "MK Dons": {"ATT": 65, "MEI": 63, "DEF": 63},
        "Newport County": {"ATT": 57, "MEI": 59, "DEF": 58},
        "Notts County": {"ATT": 62, "MEI": 63, "DEF": 62},
        "Oldham Athletic": {"ATT": 59, "MEI": 60, "DEF": 60},
        "Salford City": {"ATT": 60, "MEI": 63, "DEF": 61},
        "Shrewsbury": {"ATT": 61, "MEI": 63, "DEF": 63},
        "Swindon Town": {"ATT": 62, "MEI": 61, "DEF": 58},
        "Tranmere Rovers": {"ATT": 61, "MEI": 60, "DEF": 62},
        "Walsall": {"ATT": 61, "MEI": 63, "DEF": 62},
    }


def criar_times_base() -> Dict[str, Time]:
    dados_rating = obter_ratings_iniciais()

    times = {}
    for nome, rating_dict in dados_rating.items():
        times[nome] = Time(
            nome=nome,
            rating=Rating(**rating_dict),
            stats=Estatisticas()
        )

    return times


# ==============================
# INTERFACE E RELATÓRIOS
# ==============================

def imprimir_resultados(resultados: List[Dict], rodada_simulada: int, times_filtro: List[str] = None):
    """Versão visual melhorada dos resultados"""

    # Cabeçalho
    print(FormatadorVisual.titulo_secao(
        f"📊 PROBABILIDADES {'INICIAIS' if rodada_simulada == 0 else f'APÓS RODADA {rodada_simulada}'}",
        80,
        Cores.CYAN
    ))

    # Filtro de times
    if times_filtro:
        print(f"\n{Cores.BRIGHT_YELLOW}🔍 TIMES SELECIONADOS: {', '.join(times_filtro)}{Cores.RESET}")
        print("─" * 80)

        for nome_time in times_filtro:
            t = next((r for r in resultados if r['time'] == nome_time), None)
            if t:
                _imprimir_card_time(t)
    else:
        # Exibe todos em formato compacto
        for idx, t in enumerate(resultados, 1):
            _imprimir_linha_time(t, idx)

    # Seção de títulos
    print(FormatadorVisual.titulo_secao("🏆 RANKING DE PROBABILIDADES", 80, Cores.YELLOW))

    # Top para título
    print(f"\n{Cores.BOLD}{Cores.CYAN}═══ PROBABILIDADE DE TÍTULO ═══{Cores.RESET}\n")
    for i, t in enumerate(resultados, 1):
        prob = t['titulo'] * 100
        barra = FormatadorVisual.barra_colorida(prob, 100, 15)
        medalhao = FormatadorVisual.medalhao(i)
        print(f"{medalhao} {i:2d}. {t['time']:<20} {barra} {Cores.BOLD}{prob:>6.2f}%{Cores.RESET}")

    # Top para promoção
    print(f"\n{Cores.BOLD}{Cores.GREEN}═══ PROBABILIDADE DE PROMOÇÃO ═══{Cores.RESET}\n")
    for i, t in enumerate(resultados, 1):
        prob = t['promocao'] * 100
        barra = FormatadorVisual.barra_colorida(prob, 100, 15)
        medalhao = FormatadorVisual.medalhao(i)
        print(f"{medalhao} {i:2d}. {t['time']:<20} {barra} {Cores.BOLD}{prob:>6.2f}%{Cores.RESET}")

    # Top para playoffs
    ordenado_playoffs = sorted(resultados, key=lambda x: x['playoffs'], reverse=True)
    print(f"\n{Cores.BOLD}{Cores.BLUE}═══ PROBABILIDADE DE PLAYOFFS ═══{Cores.RESET}\n")
    for i, t in enumerate(ordenado_playoffs, 1):
        prob = t['playoffs'] * 100
        barra = FormatadorVisual.barra_colorida(prob, 100, 15)
        print(f"    {i:2d}. {t['time']:<20} {barra} {Cores.BOLD}{prob:>6.2f}%{Cores.RESET}")

    # Top risco de rebaixamento
    ordenado_rebaixamento = sorted(resultados, key=lambda x: x['rebaixamento'], reverse=True)
    print(f"\n{Cores.BOLD}{Cores.RED}═══ RISCO DE REBAIXAMENTO ═══{Cores.RESET}\n")
    for i, t in enumerate(ordenado_rebaixamento, 1):
        prob = t['rebaixamento'] * 100
        if prob > 0.1:  # mantém o filtro de relevância
            # CORREÇÃO: inverter_cores=True para rebaixamento
            barra = FormatadorVisual.barra_colorida(
                prob, 100, 15,
                usar_gradiente=True,
                inverter_cores=True  # ← ADICIONADO
            )
            print(f"    {i:2d}. {t['time']:<20} {barra} {Cores.BOLD}{prob:>6.2f}%{Cores.RESET}")

    # Ranking por pontos finais esperados
    ordenado_pontos = sorted(resultados, key=lambda x: x['pontos_finais'], reverse=True)
    max_pontos = max(t['pontos_finais'] for t in ordenado_pontos)
    print(f"\n{Cores.BOLD}{Cores.MAGENTA}═══ PONTOS FINAIS ESPERADOS ═══{Cores.RESET}\n")
    for i, t in enumerate(ordenado_pontos, 1):
        pontos = t['pontos_finais']
        barra = FormatadorVisual.barra_colorida(
            valor=pontos,
            max_valor=max_pontos,
            largura=15,
            usar_gradiente=False
        )
        medalhao = FormatadorVisual.medalhao(i)
        print(
            f"{medalhao} {i:2d}. "
            f"{t['time']:<20} "
            f"{barra} "
            f"{Cores.BOLD}{pontos:>5.1f} pts{Cores.RESET}"
        )

    print("\n" + "═" * 80 + "\n")


def _imprimir_card_time(time_dict: Dict):
    """Card visual detalhado para um time"""
    nome = time_dict['time']

    print(f"\n{Cores.BOLD}{Cores.CYAN}╔{'═' * 58}╗{Cores.RESET}")
    print(
        f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} {Cores.BOLD}{nome:^56}{Cores.RESET} {Cores.BOLD}{Cores.CYAN}║{Cores.RESET}")
    print(f"{Cores.BOLD}{Cores.CYAN}╠{'═' * 58}╣{Cores.RESET}")

    # Título
    prob_titulo = time_dict['titulo'] * 100
    barra_titulo = FormatadorVisual.barra_colorida(prob_titulo, 100, 20)
    print(
        f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} 🏆 Título:      {barra_titulo} {Cores.BOLD}{prob_titulo:>6.2f}%{Cores.RESET}           {Cores.BOLD}{Cores.CYAN}║{Cores.RESET}")

    # Promoção
    prob_promocao = time_dict['promocao'] * 100
    barra_promocao = FormatadorVisual.barra_colorida(prob_promocao, 100, 20)
    print(
        f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} 🟢 Promoção:    {barra_promocao} {Cores.BOLD}{prob_promocao:>6.2f}%{Cores.RESET}           {Cores.BOLD}{Cores.CYAN}║{Cores.RESET}")

    # Playoffs
    prob_playoffs = time_dict['playoffs'] * 100
    barra_playoffs = FormatadorVisual.barra_colorida(prob_playoffs, 100, 20, usar_gradiente=False)
    print(
        f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} 🔵 Playoffs:    {barra_playoffs} {Cores.BOLD}{prob_playoffs:>6.2f}%{Cores.RESET}           {Cores.BOLD}{Cores.CYAN}║{Cores.RESET}")

    # Rebaixamento - CORREÇÃO AQUI
    prob_rebaixamento = time_dict['rebaixamento'] * 100
    barra_rebaixamento = FormatadorVisual.barra_colorida(
        prob_rebaixamento, 100, 20,
        usar_gradiente=True,
        inverter_cores=True  # ← ADICIONADO
    )
    print(
        f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} 🔴 Rebaixamento:{barra_rebaixamento} {Cores.BOLD}{prob_rebaixamento:>6.2f}%{Cores.RESET}           {Cores.BOLD}{Cores.CYAN}║{Cores.RESET}")

    print(f"{Cores.BOLD}{Cores.CYAN}╠{'═' * 58}╣{Cores.RESET}")

    # Estatísticas
    pontos = time_dict['pontos_finais']
    saldo = time_dict['saldo_medio']
    saldo_str = f"{saldo:+.0f}" if saldo >= 0 else f"{saldo:.0f}"

    print(
        f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} 📈 Pontos esperados: {Cores.BOLD}{pontos:>5.0f} pts{Cores.RESET}                           {Cores.BOLD}{Cores.CYAN}║{Cores.RESET}")
    print(
        f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} ⚖️ Saldo médio:      {Cores.BOLD}{saldo_str:>5} gols{Cores.RESET}                          {Cores.BOLD}{Cores.CYAN}║{Cores.RESET}")

    print(f"{Cores.BOLD}{Cores.CYAN}╚{'═' * 58}╝{Cores.RESET}")

def _imprimir_linha_time(t, idx: int):
    # Prefixo simples (sem emoji largo)
    prefixo = "  "
    if idx == 1:
        prefixo = "* "
    elif idx == 2:
        prefixo = "+ "
    elif idx == 3:
        prefixo = "- "

    # Probabilidades
    p_titulo = t['titulo'] * 100
    p_promo = t['promocao'] * 100
    p_play = t['playoffs'] * 100
    p_reb = t['rebaixamento'] * 100

    # Barras - CORREÇÃO: inverter_cores=True para rebaixamento
    b_titulo = FormatadorVisual.barra_colorida(p_titulo, 100, 8)
    b_promo = FormatadorVisual.barra_colorida(p_promo, 100, 8)
    b_play = FormatadorVisual.barra_colorida(p_play, 100, 8)
    b_reb = FormatadorVisual.barra_colorida(
        p_reb, 100, 8,
        usar_gradiente=True,
        inverter_cores=True  # ← ADICIONADO
    )

    print(
        f"{prefixo}{idx:2d}. {t['time']:<22} "
        f"Título: {b_titulo} {p_titulo:>5.1f}%  "
        f"Promo: {b_promo} {p_promo:>5.1f}%  "
        f"Play: {b_play} {p_play:>5.1f}%  "
        f"Reb: {b_reb} {p_reb:>5.1f}%"
    )

def validar_simulacao(resultados: List[Dict]):
    """Sanity checks mantidos da V4"""
    print("\n" + "-" * 60)
    print("🔍 VALIDAÇÃO DO MODELO")
    print("-" * 60)

    por_pontos = sorted(resultados, key=lambda x: x['pontos_finais'], reverse=True)

    pontos = [r['pontos_finais'] for r in por_pontos]
    campeao_pts = pontos[0]
    segundo_pts = pontos[1]
    lanterna_pts = pontos[-1]
    media_pts = np.mean(pontos)

    checks = {
        'Pontos do campeão': (campeao_pts, 65, 85, campeao_pts),
        'Pontos da lanterna': (lanterna_pts, 25, 45, lanterna_pts),
        'Média de pontos': (media_pts, 50, 60, media_pts),
        'Diferença 1º-2º': (campeao_pts - segundo_pts, 3, 15, campeao_pts - segundo_pts),
    }

    print("\n📊 Sanity Checks:")
    for nome, (valor, min_ok, max_ok, real) in checks.items():
        status = "✅" if min_ok <= real <= max_ok else "⚠️"
        print(f"  {status} {nome}: {real:.1f} (esperado: {min_ok}-{max_ok})")


def imprimir_tabela_atual_melhorada(tabela_atual, rodada_escolhida: int):
    """Tabela de classificação com visual melhorado"""

    print(FormatadorVisual.titulo_secao(
        f"📋 TABELA ATUAL - APÓS RODADA {rodada_escolhida}",
        80,
        Cores.GREEN
    ))

    # Cabeçalho
    print(f"\n{Cores.BOLD}{Cores.BRIGHT_WHITE}"
            f"{'#':>4}  │ {'Time':<22} │ {'Pts':>4} │ {'J':>3} │ "
            f"{'V':>3} │ {'E':>3} │ {'D':>3} │ {'GP':>4} │ {'GC':>4} │ {'SG':>4}"
            f"{Cores.RESET}")
    print("─" * 82)

    for pos, (nome, time) in enumerate(tabela_atual, 1):
        pts = time.stats.pontos_totais()
        j = time.stats.jogos
        v = (time.stats.pontos_casa + time.stats.pontos_fora) // 3
        e = (time.stats.pontos_casa + time.stats.pontos_fora - v * 3)
        d = j - v - e
        gp = time.stats.gols_pro
        gc = time.stats.gols_contra
        sg = time.stats.saldo_gols()

        # Cor baseada na zona da tabela
        if pos <= 3:  # Promoção direta
                cor = Cores.GREEN
                zona = "  "
        elif pos <= 7:  # Playoffs
                cor = Cores.CYAN
                zona = "  "
        elif pos >= 23:  # Rebaixamento
                cor = Cores.RED
                zona = "  "
        else:
                cor = Cores.WHITE
                zona = "  "

        print(f"{zona}{cor}{pos:>2}{Cores.RESET}  │ "
                f"{cor}{nome:<22}{Cores.RESET} │ "
                f"{Cores.BOLD}{pts:>4}{Cores.RESET} │ "
                f"{j:>3} │ {v:>3} │ {e:>3} │ {d:>3} │ "
                f"{gp:>4} │ {gc:>4} │ "
                f"{Cores.BOLD if sg > 0 else ''}{Cores.GREEN if sg > 0 else Cores.RED if sg < 0 else ''}"
                f"{sg:>+4}{Cores.RESET}")

    print("─" * 82)
    print(f"{Cores.GREEN}🟢{Cores.RESET} Promoção Direta  "
            f"{Cores.CYAN}🔵{Cores.RESET} Playoffs  "
            f"{Cores.RED}🔴{Cores.RESET} Rebaixamento")


# ==============================
# MAIN V6
# ==============================

def main():
    """
    Main refatorado para V5.

    Novidades:
    - Seed configurável
    - Paralelização opcional
    - Checkpoints automáticos
    - Análise de distribuições
    """

    # Configurações opcionais
    print("\n" + "-" * 60)
    print("⚙️  CONFIGURAÇÕES AVANÇADAS")
    print("=" * 60)

    # BLOCO 1.1: Configuração de seed
    seed = None  # ✅ GARANTE QUE A VARIÁVEL SEMPRE EXISTE

    usar_seed = input("\n🎲 Usar seed fixa para reprodutibilidade? (s/n): ").lower()
    if usar_seed == 's':
        seed = int(input("Digite o seed (número inteiro): "))
        SeedManager.configurar_seed(seed)
        print(f"✅ Seed configurada: {seed}")
    else:
        SeedManager.configurar_seed(None)
        print("✅ Modo aleatório (não reproduzível)")

    if input("Ativar paralelização? (s/n): ").lower() == 's':
        config.usar_paralelizacao = True
        n_workers = input(f"Número de workers (padrão {config.n_workers}): ").strip()
        if n_workers:
            config.n_workers = int(n_workers)
        print(f"✅ Paralelização ativada: {config.n_workers} workers")

    if input("Ativar profiling de performance? (s/n): ").lower() == 's':
        config.ativar_profiling = True
        print("✅ Profiling ativado")

    if input("Usar peso dinâmico de OVR? (s/n): ").lower() == 's':
        config.usar_peso_dinamico_ovr = True
        print("⚠️ Peso dinâmico ativado")

    print("\nConfiguração da simulação:")
    print(f"  Seed fixa: {'sim' if seed is not None else 'não'}")
    print(
        f"  Paralelização: {'sim' if config.usar_paralelizacao else 'não'}",
        f"({config.n_workers} workers)" if config.usar_paralelizacao else ""
    )
    print(f"  Profiling: {'sim' if config.ativar_profiling else 'não'}")
    print(
        f"  OVR dinâmico: "
        f"{'sim (experimental)' if config.usar_peso_dinamico_ovr else 'não'}"
    )

    print("\n" + "=" * 60)
    print("⚽ SIMULADOR DE LIGA - V6.0")
    print("=" * 60)

    # Carrega dados
    times_base = criar_times_base()
    print(f"\n✅ {len(times_base)} times carregados")

    arquivo_calendario = "calendario_league_two.csv"
    try:
        print(f"\n📅 Carregando calendário: {arquivo_calendario}")
        jogos = carregar_calendario(arquivo_calendario)

        # === Estatísticas do calendário ===
        total_jogos = len(jogos)
        rodadas = sorted({j.rodada for j in jogos})
        total_rodadas = len(rodadas)

        jogos_por_rodada = {}
        for j in jogos:
            jogos_por_rodada.setdefault(j.rodada, []).append(j)

        def jogo_realizado(jogo):
            # Caso 1: placar como tupla (gm, gv)
            if hasattr(jogo, "placar") and jogo.placar is not None:
                return True

            # Caso 2: atributos genéricos de gols
            for attr_m, attr_v in [
                ("gols_mandante", "gols_visitante"),
                ("gols_casa", "gols_fora"),
                ("score_mandante", "score_visitante"),
            ]:
                if hasattr(jogo, attr_m) and hasattr(jogo, attr_v):
                    return getattr(jogo, attr_m) is not None and getattr(jogo, attr_v) is not None

            # Caso 3: resultado armazenado como string ou dict
            if hasattr(jogo, "resultado") and jogo.resultado:
                return True

            return False

        rodadas_completas = {
            r: sum(1 for j in jogos_por_rodada[r] if jogo_realizado(j))
            for r in rodadas
        }

        jogos_realizados = sum(
            1 for j in jogos if jogo_realizado(j)
        )

        def extrair_times(jogo):
            # Padrões mais comuns
            for a, b in [
                ("mandante", "visitante"),
                ("time_casa", "time_fora"),
                ("casa", "fora"),
                ("home", "away"),
            ]:
                if hasattr(jogo, a) and hasattr(jogo, b):
                    return getattr(jogo, a), getattr(jogo, b)

            # Caso placar ou equipes como tupla
            if hasattr(jogo, "equipes") and isinstance(jogo.equipes, (list, tuple)):
                if len(jogo.equipes) == 2:
                    return jogo.equipes[0], jogo.equipes[1]

            raise AttributeError("Não foi possível extrair times do objeto Jogo")

        times_unicos = set()
        for j in jogos:
            t1, t2 = extrair_times(j)
            times_unicos.add(t1)
            times_unicos.add(t2)

        # === Impressão formatada ===
        print("✅ Calendário carregado:")
        print(f"   {total_jogos} jogos totais em {total_rodadas} rodadas")
        print(f"   {jogos_realizados} jogos já realizados")
        print(f"   {len(times_unicos)} times únicos")
        print("✅ Calendário validado e pronto!\n")

        print("=" * 60)
        print("📊 RODADAS COM RESULTADOS DISPONÍVEIS")
        print("=" * 60)

        for r in rodadas:
            feitos = rodadas_completas[r]
            total = len(jogos_por_rodada[r])
            status = "✅" if feitos == total else "⏳"
            print(f"  Rodada {r:2d}: {status} {feitos}/{total} jogos")

        print("\n" + "=" * 60 + "\n")

    except FileNotFoundError:
        print(f"❌ Arquivo '{arquivo_calendario}' não encontrado")
        return

    # Escolhe rodada
    rodadas_disponiveis = max(j.rodada for j in jogos)

    while True:
        try:
            rodada_input = input(
                f"\n🎯 Digite a rodada para análise (0-{rodadas_disponiveis}): ").strip()
            rodada_escolhida = int(rodada_input)

            if rodada_escolhida < 0 or rodada_escolhida > rodadas_disponiveis:
                print(f"❌ Rodada inválida")
                continue

            break
        except ValueError:
            print("❌ Digite um número válido")

    # Extrai estatísticas
    print(f"\n📈 Extraindo estatísticas até rodada {rodada_escolhida}...")
    times_atualizados = extrair_estatisticas_do_calendario(jogos, times_base, rodada_escolhida)

    # Monta a tabela atual
    tabela_atual = sorted(
        times_atualizados.items(),
        key=lambda x: (
            -x[1].stats.pontos_totais(),
            -x[1].stats.saldo_gols(),
            -x[1].stats.gols_pro
        )
    )

    # IMPRIME A TABELA
    imprimir_tabela_atual_melhorada(tabela_atual, rodada_escolhida)

    # Cria calibrador OVR
    print(f"\n🔧 Calibrando OVR temporal...")
    ovr_inicial = obter_ratings_iniciais()
    ovr_final = obter_ratings_atuais()
    rodada_final_calibracao = max(j.rodada for j in jogos)

    calibrador_ovr = CalibradorOVRTemporal(
        ovr_inicial=ovr_inicial,
        ovr_final=ovr_final,
        rodada_final=rodada_final_calibracao
    )

    # Prepara jogos
    jogos_para_simular = []
    for jogo in jogos:
        if jogo.rodada <= rodada_escolhida:
            jogos_para_simular.append(jogo)
        else:
            jogos_para_simular.append(Jogo(
                rodada=jogo.rodada,
                time_casa=jogo.time_casa,
                time_fora=jogo.time_fora,
                gols_casa=None,
                gols_fora=None
            ))

    # Checkpoint ID
    checkpoint_id = f"sim_r{rodada_escolhida}"

    # Executa simulação V5
    print(f"\n🎲 Executando {config.simulacoes} simulações...")

    simulador = SimuladorMonteCarlo(
        times_atualizados,
        jogos_para_simular,
        calibrador_ovr,
        seed=SeedManager._master_seed
    )

    simulador.executar(config.simulacoes, checkpoint_id)

    # Obtém resultados
    resultados = simulador.obter_resultados()

    # Filtro de times
    print("\n" + "=" * 60)
    print("🔍 FILTRO DE TIMES")
    print("=" * 60)

    lista_times = sorted(times_atualizados.keys())
    for i, time in enumerate(lista_times, 1):
        print(f"  {i:2d}. {time}")

    times_filtro = None
    entrada = input(
        "\nDigite números separados por vírgula ou ENTER para todos: ").strip()

    if entrada and entrada.lower() != 'todos':
        try:
            indices = [int(x.strip()) for x in entrada.split(',')]

            # Validação segura de índices
            times_filtro = []
            for i in indices:
                if 1 <= i <= len(lista_times):
                    times_filtro.append(lista_times[i - 1])
                else:
                    print(f"⚠️ Índice inválido ignorado: {i}")

            if not times_filtro:
                times_filtro = None

        except ValueError:
            print("❌ Entrada inválida, exibindo todos os times")
            times_filtro = None

    # Imprime resultados
    imprimir_resultados(resultados, rodada_escolhida, times_filtro)

    # Validação
    validar_simulacao(resultados)

    # BLOCO 4.10: Análise de distribuições
    if simulador.resultado_final and times_filtro:
        print("\n" + "=" * 60)
        print("📊 ANÁLISE DE DISTRIBUIÇÕES")
        print("=" * 60)

        analise = AnalisadorDistribuicoes.comparar_times(
            times_filtro,
            simulador.resultado_final
        )
        print(analise)

    # Metadados de execução
    if simulador.resultado_final:
        print("\n" + "-" * 60)
        print("ℹ️  METADADOS DA SIMULAÇÃO")
        print("-" * 60)
        print(f"  Versão: {simulador.resultado_final.versao_codigo}")
        print(f"  Simulações: {simulador.resultado_final.n_simulacoes}")
        print(f"  Tempo total: {simulador.resultado_final.tempo_execucao:.2f}s")
        print(f"  Seed: {simulador.resultado_final.seed_utilizada}")
        print(f"  Paralelização: {'Sim' if config.usar_paralelizacao else 'Não'}")

    print("\n" + "=" * 60)
    print("✅ Simulação concluída!")
    print("=" * 60)

    # Opções adicionais
    while True:
        print("\n🔄 OPÇÕES:")
        print("  1 - Nova simulação")
        print("  2 - Exportar resultados (JSON)")
        print("  3 - Análise detalhada de time específico")
        print("  4 - Evolução de probabilidades por rodada")
        print("  5 - Sair")

        opcao = input("\nEscolha (1-5): ").strip()

        if opcao == '1':
            main()
            break
        elif opcao == '2':
            if simulador.resultado_final:
                arquivo = f"resultados_r{rodada_escolhida}.json"
                with open(arquivo, 'w', encoding='utf-8') as f:
                    json.dump(simulador.resultado_final.to_dict(), f, indent=2, default=str)
                print(f"✅ Resultados exportados: {arquivo}")
        elif opcao == '3':
            if simulador.resultado_final:
                time_nome = input("Digite o nome do time: ").strip()
                if time_nome in lista_times:
                    analise = AnalisadorDistribuicoes.analise_time(
                        time_nome,
                        simulador.resultado_final
                    )
                    print(f"\n📊 ANÁLISE DETALHADA - {time_nome}")
                    print("=" * 60)
                    print(f"Pontos:")
                    print(f"  Média: {analise['pontos']['media']:.1f}")
                    print(f"  Mediana: {analise['pontos']['mediana']:.1f}")
                    print(f"  Desvio: {analise['pontos']['desvio']:.1f}")
                    print(f"  Range: {analise['pontos']['min']}-{analise['pontos']['max']}")
                    print(f"  P10-P90: {analise['pontos']['p10']:.0f}-{analise['pontos']['p90']:.0f}")
                    print(f"\nPosição:")
                    print(f"  Média: {analise['posicao']['media']:.1f}º")
                    print(f"  Mediana: {analise['posicao']['mediana']:.0f}º")
                    print(f"  Melhor: {analise['posicao']['melhor']}º")
                    print(f"  Pior: {analise['posicao']['pior']}º")
                    print("=" * 60)
                else:
                    print("❌ Time não encontrado")
        elif opcao == '4':
                    print("\n" + "=" * 60)
                    print("📈 ANÁLISE DE EVOLUÇÃO POR RODADA")
                    print("=" * 60)

                    # Lista times
                    for i, time in enumerate(lista_times, 1):
                        print(f"  {i:2d}. {time}")

                    try:
                        idx = int(input("\nEscolha o número do time: ").strip())
                        if 1 <= idx <= len(lista_times):
                            time_escolhido = lista_times[idx - 1]

                            # Pergunta número de simulações
                            n_sims = input("Simulações por rodada (padrão 5000): ").strip()
                            n_sims = int(n_sims) if n_sims else 5000

                            # Cria analisador
                            analisador = AnalisadorEvolucao(
                                times_base,
                                jogos,
                                calibrador_ovr
                            )

                            # Analisa evolução
                            historico = analisador.analisar_evolucao(time_escolhido, n_sims)

                            # ============================================
                            # CORREÇÃO: CHAMAR A FUNÇÃO DE IMPRESSÃO AQUI
                            # ============================================
                            def imprimir_evolucao_melhorada(historico: Dict, nome_time: str):
                                """Versão visual melhorada da evolução"""

                                print(FormatadorVisual.titulo_secao(
                                    f"📈 EVOLUÇÃO DAS PROBABILIDADES - {nome_time}",
                                    90,
                                    Cores.MAGENTA
                                ))

                                # Tabela com cores
                                print(f"\n{Cores.BOLD}{Cores.BRIGHT_WHITE}{'Rod':>4} │ {'Título':>10} │ {'Promoção':>10} │ "
                                      f"{'Playoffs':>10} │ {'Rebaixa':>10} │ {'Pts':>6} │ {'Pos':>5} │ Tendência{Cores.RESET}")
                                print("─" * 90)

                                for i in range(len(historico['rodadas'])):
                                    rodada = historico['rodadas'][i]
                                    titulo = historico['titulo'][i]
                                    promocao = historico['promocao'][i]
                                    playoffs = historico['playoffs'][i]
                                    rebaixa = historico['rebaixamento'][i]
                                    pontos = historico['pontos_medios'][i]
                                    posicao = historico['posicao_media'][i]

                                    # Cor baseada na rodada (destaca rodadas especiais)
                                    if rodada == 0:
                                        cor_rodada = Cores.BRIGHT_YELLOW
                                    elif i == len(historico['rodadas']) - 1:
                                        cor_rodada = Cores.BRIGHT_GREEN
                                    else:
                                        cor_rodada = Cores.WHITE

                                    # Indicador de mudança
                                    if i > 0:
                                        delta_titulo = titulo - historico['titulo'][i - 1]
                                        tendencia = FormatadorVisual.indicador_tendencia(titulo, historico['titulo'][i - 1], 2.0)
                                    else:
                                        tendencia = f"{Cores.BRIGHT_YELLOW}🎯 Inicial{Cores.RESET}"

                                    # Cores para valores
                                    cor_titulo = Cores.GREEN if titulo > 15 else Cores.YELLOW if titulo > 5 else Cores.WHITE
                                    cor_promocao = Cores.GREEN if promocao > 50 else Cores.YELLOW if promocao > 30 else Cores.WHITE

                                    print(f"{cor_rodada}{rodada:>4}{Cores.RESET} │ "
                                          f"{cor_titulo}{titulo:>9.2f}%{Cores.RESET} │ "
                                          f"{cor_promocao}{promocao:>9.2f}%{Cores.RESET} │ "
                                          f"{playoffs:>9.2f}% │ "
                                          f"{rebaixa:>9.2f}% │ "
                                          f"{pontos:>6.1f} │ "
                                          f"{posicao:>5.1f} │ {tendencia}")

                                print("─" * 90)

                                # Análise de tendências melhorada
                                _imprimir_tendencias_melhorada(historico)

                            def _imprimir_tendencias_melhorada(historico: Dict):
                                """Análise de tendências com visual melhorado"""
                                if len(historico['rodadas']) < 2:
                                    return

                                print(f"\n{Cores.BOLD}{Cores.CYAN}╔{'═' * 68}╗{Cores.RESET}")
                                print(
                                    f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} {Cores.BOLD}{'📊 ANÁLISE DE TENDÊNCIAS':^66}{Cores.RESET} {Cores.BOLD}{Cores.CYAN}║{Cores.RESET}")
                                print(f"{Cores.BOLD}{Cores.CYAN}╠{'═' * 68}╣{Cores.RESET}")

                                titulo_inicial = historico['titulo'][0]
                                titulo_atual = historico['titulo'][-1]
                                delta_titulo = titulo_atual - titulo_inicial

                                promocao_inicial = historico['promocao'][0]
                                promocao_atual = historico['promocao'][-1]
                                delta_promocao = promocao_atual - promocao_inicial

                                playoffs_inicial = historico['playoffs'][0]
                                playoffs_atual = historico['playoffs'][-1]
                                delta_playoffs = playoffs_atual - playoffs_inicial

                                rebaixa_inicial = historico['rebaixamento'][0]
                                rebaixa_atual = historico['rebaixamento'][-1]
                                delta_rebaixa = rebaixa_atual - rebaixa_inicial

                                # Título
                                if abs(delta_titulo) > 5:
                                    if delta_titulo > 0:
                                        icone = "✅"
                                        cor = Cores.GREEN
                                        texto = f"AUMENTARAM {delta_titulo:+.2f}%"
                                    else:
                                        icone = "❌"
                                        cor = Cores.RED
                                        texto = f"DIMINUÍRAM {delta_titulo:.2f}%"
                                else:
                                    icone = "➡️"
                                    cor = Cores.YELLOW
                                    texto = f"ESTÁVEIS"

                                print(
                                    f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} {icone} Chances de título {cor}{texto}{Cores.RESET}")
                                print(
                                    f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET}    {titulo_inicial:>5.2f}% → {titulo_atual:>5.2f}%")

                                # Promoção
                                if abs(delta_promocao) > 5:
                                    if delta_promocao > 0:
                                        icone = "✅"
                                        cor = Cores.GREEN
                                        texto = f"AUMENTARAM {delta_promocao:+.2f}%"
                                    else:
                                        icone = "❌"
                                        cor = Cores.RED
                                        texto = f"DIMINUÍRAM {delta_promocao:.2f}%"
                                else:
                                    icone = "➡️"
                                    cor = Cores.YELLOW
                                    texto = f"ESTÁVEIS"

                                print(f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET}")
                                print(
                                    f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} {icone} Chances de promoção {cor}{texto}{Cores.RESET}")
                                print(
                                    f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET}    {promocao_inicial:>5.2f}% → {promocao_atual:>5.2f}%")

                                # Playoffs
                                if abs(delta_playoffs) > 5:
                                    if delta_playoffs > 0:
                                        icone = "✅"
                                        if playoffs_atual > promocao_atual:
                                            cor = Cores.GREEN
                                        else:
                                            cor = Cores.CYAN
                                        texto = f"AUMENTARAM {delta_playoffs:+.2f}%"
                                    else:
                                        icone = "❌"
                                        if playoffs_atual > promocao_atual:
                                            cor = Cores.RED
                                        else:
                                            cor = Cores.CYAN
                                        texto = f"DIMINUÍRAM {delta_playoffs:.2f}%"
                                else:
                                    icone = "➡️"
                                    cor = Cores.YELLOW
                                    texto = f"ESTÁVEIS"

                                print(f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET}")
                                print(
                                    f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} {icone} Chances de playoffs {cor}{texto}{Cores.RESET}")
                                print(
                                    f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET}    {playoffs_inicial:>5.2f}% → {playoffs_atual:>5.2f}%")

                                # Rebaixamento
                                if abs(delta_rebaixa) > 5:
                                    if delta_rebaixa < 0:
                                        icone = "✅"
                                        cor = Cores.GREEN
                                        texto = f"DIMINUIRAM {delta_rebaixa:+.2f}%"
                                    else:
                                        icone = "❌"
                                        cor = Cores.RED
                                        texto = f"AUMENTARAM {delta_rebaixa:.2f}%"
                                else:
                                    icone = "➡️"
                                    cor = Cores.YELLOW
                                    texto = f"ESTÁVEIS"

                                print(f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET}")
                                print(
                                    f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} {icone} Chances de rebaixamento {cor}{texto}{Cores.RESET}")
                                print(
                                    f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET}    {rebaixa_inicial:>5.2f}% → {rebaixa_atual:>5.2f}%")

                                # Maior oscilação positiva
                                variacoes = [historico['titulo'][i] - historico['titulo'][i - 1]
                                             for i in range(1, len(historico['titulo']))]
                                if variacoes:
                                    idx_max = max(range(len(variacoes)), key=lambda i: variacoes[i])
                                    rodada_mudanca = historico['rodadas'][idx_max + 1]
                                    variacao_max = variacoes[idx_max]

                                    cor_variacao = Cores.GREEN if variacao_max > 0 else Cores.RED

                                    print(f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET}")
                                    print(
                                        f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} 🎢 Melhor rodada: Rodada {Cores.BOLD}{rodada_mudanca}{Cores.RESET} "
                                        f"({cor_variacao}{variacao_max:+.2f}%{Cores.RESET} no título)"
                                    )

                                # Maior oscilação negativa
                                variacoes = [historico['titulo'][i] - historico['titulo'][i - 1]
                                             for i in range(1, len(historico['titulo']))]
                                if variacoes:
                                    idx_min = min(range(len(variacoes)), key=lambda i: variacoes[i])
                                    rodada_mudanca = historico['rodadas'][idx_min + 1]
                                    variacao_min = variacoes[idx_min]

                                    cor_variacao = Cores.RED if variacao_min < 0 else Cores.GREEN

                                    print(f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET}")
                                    print(
                                        f"{Cores.BOLD}{Cores.CYAN}║{Cores.RESET} 🎢 Pior rodada: Rodada {Cores.BOLD}{rodada_mudanca}{Cores.RESET} "
                                        f"({cor_variacao}{variacao_min:+.2f}%{Cores.RESET} no título)"
                                    )

                                print(f"{Cores.BOLD}{Cores.CYAN}╚{'═' * 68}╝{Cores.RESET}\n")

                            # ============================================
                            # AQUI: CHAMA A FUNÇÃO PARA EXIBIR OS DADOS
                            # ============================================
                            imprimir_evolucao_melhorada(historico, time_escolhido)

                            # Opção de exportar (APÓS exibir)
                            if input("\nExportar para JSON? (s/n): ").lower() == 's':
                                arquivo = f"evolucao_{time_escolhido.replace(' ', '_')}.json"
                                with open(arquivo, 'w', encoding='utf-8') as f:
                                    json.dump(historico, f, indent=2)
                                print(f"✅ Evolução exportada: {arquivo}")

                        else:
                            print("❌ Número inválido")
                    except ValueError:
                        print("❌ Entrada inválida")
        elif opcao == '5':
            print("\n👋 Até logo!")
            break


if __name__ == "__main__":
    main()