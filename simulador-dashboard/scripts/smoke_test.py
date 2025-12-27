from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from simulador import (
    CalibradorOVRTemporal,
    Jogo,
    SimuladorMonteCarlo,
    config,
    criar_times_base,
    extrair_estatisticas_do_calendario,
    obter_ratings_atuais,
    obter_ratings_iniciais,
    carregar_calendario,
)


def main() -> None:
    times_base = criar_times_base()
    calendario_path = PROJECT_DIR / "data" / "calendario_league_two.csv"
    jogos = carregar_calendario(calendario_path)

    rodada_escolhida = 0
    times_atualizados = extrair_estatisticas_do_calendario(
        jogos,
        times_base,
        rodada_escolhida,
    )

    calibrador_ovr = CalibradorOVRTemporal(
        ovr_inicial=obter_ratings_iniciais(),
        ovr_final=obter_ratings_atuais(),
        rodada_final=max(jogo.rodada for jogo in jogos),
    )

    jogos_para_simular = []
    for jogo in jogos:
        if jogo.rodada <= rodada_escolhida:
            jogos_para_simular.append(jogo)
        else:
            jogos_para_simular.append(
                Jogo(
                    rodada=jogo.rodada,
                    time_casa=jogo.time_casa,
                    time_fora=jogo.time_fora,
                    gols_casa=None,
                    gols_fora=None,
                )
            )

    config.atualizar_progresso_a_cada = 50
    simulador = SimuladorMonteCarlo(
        times_atualizados,
        jogos_para_simular,
        calibrador_ovr,
        seed=42,
    )
    simulador.executar(50)

    if simulador.resultado_final is None:
        raise RuntimeError("Simulação não produziu resultado final")

    print("OK")


if __name__ == "__main__":
    main()
