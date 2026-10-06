# Dados

Os dados não são distribuídos neste repositório. A telemetria foi cedida pela equipe mediante Termo de Anuência, e os condutores são identificados apenas por pseudônimo (P1, P2).

Para reproduzir os resultados, coloque aqui o arquivo `dataset_2pilotos.csv`. Para regenerá-lo com `src/01_monta_dataset.py`, coloque os CSV brutos exportados em `dados/brutos/`.

## Dicionário — `dataset_2pilotos.csv` (1.120 linhas = 112 voltas × 10 setores)

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| piloto | categórica | — | Condutor (P1, P2) |
| sessao | texto | — | Identificador da sessão |
| tipo | categórica | — | Natureza da sessão (TRP, TO, TL, Q1, R1, R2) |
| volta | texto | — | Identificador da volta (grupo da validação) |
| setor | categórica | — | T1 a T10: décimos de igual distância da volta (4.309 m) |
| tempo_setor | contínua | s | Tempo gasto no setor |
| pbrake_max | contínua | bar | Pressão máxima de freio dianteira |
| pbrake_media | contínua | bar | Pressão média de freio dianteira |
| t_freio | contínua | s | Tempo com pressão > 5 bar (amostragem de 100 Hz) |
| pbrake_taxa | contínua | bar s⁻¹ | Taxa máxima de variação da pressão de freio |
| accx_min | contínua | g | Percentil 1 da aceleração longitudinal |
| accx_max | contínua | g | Percentil 99 da aceleração longitudinal |
| rpm_max | contínua | rpm | Rotação máxima do motor |
| marcha_max | discreta | — | Marcha máxima engatada no setor |
| trocas | discreta | — | Número de trocas de marcha no setor |
| slip | contínua | km h⁻¹ | Máxima diferença de velocidade entre a roda traseira e a dianteira esquerdas |

A aceleração longitudinal é a derivada numérica da velocidade (`np.gradient`) dividida por 9,81.

## Filtros aplicados

- Uma volta é válida se cobre ao menos 98,5% do comprimento da pista, não tem velocidade abaixo de 25 km/h (exclui paradas) e dura entre 95 e 145 s.
- Um setor precisa de ao menos 20 amostras.
