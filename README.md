# Predição do tempo por setor em telemetria de automobilismo

Código e resultados do Trabalho de Conclusão de Curso do **MBA em Data Science e Analytics — USP/ESALQ**.

- **Autor:** Eder Alves da Silva
- **Orientadora:** Profa. Dra. Ana Julia Righetto
- **Texto completo:** _(inserir o link após a aprovação)_

## Objetivo

Prever o desvio do tempo de cada setor do circuito em relação à mediana do condutor naquele setor, a partir de canais de acionamento do veículo: freio, aceleração longitudinal, rotação, marcha e escorregamento. Os dados são telemetria real de dois condutores (P1 e P2) em um Porsche GT3 no Autódromo de Interlagos. São 112 voltas, divididas em 10 setores de igual distância, o que dá 1.120 observações.

## Estrutura

```
src/
  01_monta_dataset.py          telemetria bruta (CSV) -> dados/dataset_2pilotos.csv
  02_avalia_10_particoes.py    Tabela 4: 6 modelos x 10 partições agrupadas por volta
  03_ajuste_hiperparametros.py Tabela 2: busca em grade com GroupKFold (5) no treino
dados/
  README.md                    dicionário de dados e política de acesso
resultados/
  tabela_final.csv             Tabela 4 (média e desvio-padrão de 10 partições)
  res_*.json, resultado_tunado.json   Tabela 2 (hiperparâmetros selecionados)
  importancia_repetida.csv     Tabela 5 (importância por permutação)
  repetido.json                Spearman por volta, matriz de confusão em tercis, RMSE por setor/condutor
  figuras/                     Figuras 1 a 8 do texto
```

## Como reproduzir

```bash
pip install -r requirements.txt
# coloque dados/dataset_2pilotos.csv (ver dados/README.md)
python src/02_avalia_10_particoes.py              # Tabela 4
python src/03_ajuste_hiperparametros.py gb 800    # Tabela 2, um modelo por vez: base, lr, arvore, rf, gb, mlp
```

## Desenho experimental

- **Variável resposta:** `tempo_setor` menos a mediana do condutor no setor.
- **Validação:** as partições são agrupadas por volta (`GroupShuffleSplit`, 25% das voltas para teste), para que setores da mesma volta não apareçam no treino e no teste ao mesmo tempo. A Tabela 4 é a média de 10 partições (sementes 1000 a 1009).
- **Ajuste de hiperparâmetros:** `GridSearchCV` com `GroupKFold(5)` dentro do treino, semente 7895.
- **Pré-processamento:** one-hot de `setor` e `tipo`. A padronização (`StandardScaler`) é aplicada só na rede neural e na regressão linear.
- **Referência:** predição nula do desvio, o que equivale à mediana por condutor-setor.

## Resultados principais (Tabela 4)

| Modelo | RMSE (s) | MAE (s) | R² | Ganho sobre a referência |
|---|---|---|---|---|
| Rede neural | 1,003 ± 0,176 | 0,659 | 0,811 | 57,4% |
| Gradient Boosting | 1,018 ± 0,211 | 0,637 | 0,806 | 57,1% |
| Random Forest | 1,175 ± 0,213 | 0,728 | 0,746 | 50,5% |
| Regressão linear múltipla | 1,326 ± 0,232 | 0,836 | 0,677 | 44,1% |
| Árvore de decisão | 1,501 ± 0,300 | 0,960 | 0,585 | 36,9% |
| Mediana por condutor-setor | 2,381 ± 0,312 | 1,446 | −0,013 | — |

Com as versões fixadas em `requirements.txt` (Python 3.12.3, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4 — as mesmas citadas no texto), `02_avalia_10_particoes.py` reproduz a Tabela 4 exatamente. Em versões mais novas do scikit-learn, o Gradient Boosting e a árvore de decisão podem variar na terceira casa decimal.

## Dados

A telemetria pertence à equipe e foi cedida mediante Termo de Anuência. Os arquivos brutos **não** são distribuídos, e os condutores aparecem apenas como P1 e P2. Veja `dados/README.md`.

## Licença

Código sob licença MIT (ver `LICENSE`). Figuras e resultados: CC BY 4.0.
