# Respira

Estimativa do nível de ansiedade a partir de hábitos de vida e sinais vitais.
Entrega 1 · Machine Learning · Projeto Final

> ⚠️ Projeto acadêmico com dados sintéticos. Não é instrumento de triagem clínica nem diagnóstico.

---

## 1. Grupo e repositório

| Campo | Preenchimento |
|---|---|
| **Projeto** | Respira |
| **Repositório** | https://github.com/`<usuario>`/modelo-machinelearning <!-- PREENCHER --> |
| **Integrantes** | <!-- PREENCHER: 3 a 4 nomes --> |

---

## 2. Resumo da proposta

| Campo | Preenchimento |
|---|---|
| **Problema** | Dado o perfil de hábitos e sinais vitais de uma pessoa, estimar seu nível de ansiedade para que ela identifique quais hábitos mais pesam e busque ajuda antes do quadro se agravar |
| **Usuário** | a própria pessoa, num autoteste, ou um serviço de bem-estar na triagem inicial |
| **Tipo e alvo** | **regressão** · alvo: `Anxiety Level (1-10)`, escala contínua de 1 a 10 · média 3,93 |
| **Dataset** | Kaggle *Social Anxiety Dataset* · 11.000 linhas × 19 colunas · **CC0 domínio público** · zero nulos, zero duplicadas |
| **Métrica** | **MAE** em pontos da escala: diz direto "o modelo erra X ponto", que é o que o usuário entende |
| **Baseline** | prever a média: MAE 1,605 · linear só com estresse: MAE 1,221 · **modelo: MAE 0,811** |
| **Riscos** | dado sintético (modelo aprende o gerador, não a ansiedade); terapia e medicação são consequência, não causa |

---

## 3. Dataset

**Link:** https://www.kaggle.com/datasets/natezhang123/social-anxiety-dataset
**Arquivo:** `enhanced_anxiety_dataset.csv`
**Licença:** [CC0 1.0 — Domínio Público](https://creativecommons.org/publicdomain/zero/1.0/). Uso livre, inclusive comercial, sem exigência de atribuição.
**Linhas × colunas:** 11.000 × 19 · **18 variáveis de entrada + 1 alvo** · nenhum valor ausente, nenhuma linha duplicada

### Dicionário das colunas de entrada

| Coluna | Tipo | Descrição |
|---|---|---|
| `Age` | int | Idade em anos |
| `Gender` | categórica | `Male`, `Female`, `Other` |
| `Occupation` | categórica | 13 profissões (`Student`, `Nurse`, `Engineer`, `Artist`…) |
| `Sleep Hours` | float | Horas de sono por noite |
| `Physical Activity (hrs/week)` | float | Horas de atividade física por semana |
| `Caffeine Intake (mg/day)` | int | Cafeína por dia, em mg |
| `Alcohol Consumption (drinks/week)` | int | Doses de álcool por semana |
| `Smoking` | binária | Fuma — `Yes` / `No` |
| `Family History of Anxiety` | binária | Histórico familiar de ansiedade |
| `Stress Level (1-10)` | int | Nível de estresse autorreferido |
| `Heart Rate (bpm)` | int | Frequência cardíaca |
| `Breathing Rate (breaths/min)` | int | Frequência respiratória |
| `Sweating Level (1-5)` | int | Nível de sudorese |
| `Dizziness` | binária | Tem tonturas |
| `Medication` | binária | Usa medicação |
| `Therapy Sessions (per month)` | int | Sessões de terapia por mês |
| `Recent Major Life Event` | binária | Evento marcante recente |
| `Diet Quality (1-10)` | int | Qualidade da alimentação |
| `Anxiety Level (1-10)` | float | **ALVO** |

Todas as 18 entradas passam no teste do formulário: são respostas que a própria pessoa fornece ou mede em casa.

> Os nomes das colunas ficam em inglês porque é assim que vêm no arquivo; traduzi-los quebraria a correspondência com o CSV.

**Variáveis mais correlacionadas com o alvo:** estresse (0,67), sessões de terapia (0,52), horas de sono (−0,49), cafeína (0,35), atividade física (−0,24).

---

## 4. Métrica e baseline

**Principal: MAE (erro absoluto médio).** O alvo é uma nota de 1 a 10, então o MAE responde na própria unidade do problema: "o modelo erra, em média, 0,8 ponto". É interpretável por quem não é da área, ao contrário do RMSE e do R². Usamos RMSE e R² como apoio — o RMSE para detectar erros grandes isolados e o R² para comparar com a literatura.

**Por que não acurácia:** o alvo é contínuo. Transformá-lo em classes jogaria fora informação e criaria um limiar arbitrário.

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| **B0** — prever sempre a média | 1,605 | 2,153 | 0,000 |
| **B1** — copiar o nível de estresse | 2,357 | 2,959 | −0,890 |
| **B2** — regressão linear só com estresse | **1,221** | 1,616 | 0,436 |
| Regressão linear completa | 0,892 | 1,126 | 0,726 |
| Gradient boosting | **0,811** | 1,007 | **0,781** |

*Divisão treino/teste 80/20, semente fixa. Treino: 8.800 · Teste: 2.200.*

**Resultado:** o melhor baseline é o B2, com MAE 1,221. O modelo chega a 0,811, uma **redução de 0,41 ponto** no erro.

**B1 falha de um jeito informativo.** A hipótese preguiçosa "ansiedade é a mesma coisa que estresse" fica **pior que prever a média** (R² negativo): as duas escalas vão de 1 a 10, mas têm médias diferentes, então copiar o valor erra por deslocamento. É por isso que o B2 existe — ele é o mesmo sinal, só que calibrado.

**Validação:** divisão estratificada 80/20 com semente fixa. Na E2, validação cruzada em 5 partições para medir a variância do resultado.

---

## 5. Esboço do fluxo

```
Formulário (Streamlit)          API (FastAPI)            Resposta
─────────────────────           ─────────────            ────────
idade, gênero, profissão   →    valida faixas       →    nota estimada (1-10)
sono, atividade física          codifica categorias      faixa: leve/moderada/alta
cafeína, álcool, fumo           predict()                3 hábitos que mais pesaram
estresse, sono, dieta                                    o quanto mudar cada um
batimentos, respiração                                   ajudaria na estimativa
tonturas, terapia, remédio                               aviso: procure um profissional
```

**Entra:** os 18 campos. **Sai:** a nota estimada com a faixa correspondente, os hábitos que mais pesaram naquele caso e uma simulação do tipo "dormir 1 hora a mais reduziria a estimativa em X" — o que torna o resultado acionável em vez de apenas informativo.

---

## 6. Riscos e limitações

- **O dado é sintético.** Zero nulos, zero duplicadas e nenhum valor impossível em 11 mil linhas não ocorre em pesquisa real. O modelo aprende as regras do gerador, não a ansiedade humana. Nenhuma conclusão daqui vale como achado clínico — mas a modelagem, a validação e a análise de erro são exatamente as mesmas.
- **Causalidade reversa:** terapia e medicação aparecem associadas a ansiedade mais alta porque quem as usa já é ansioso. São consequência, não causa, e por isso não podem ser lidas como fatores de risco. Removendo as duas, o MAE sobe só de 0,811 para 0,828 — o modelo quase não depende delas.
- **O alvo é autorreferido** e correlacionado ao estresse, que também é autorreferido. Parte do que o modelo captura pode ser o jeito de a pessoa responder, não o estado dela.
- **Notas altas são raras** (8 a 10 somam ~9% dos casos), então o modelo tende a errar mais justamente nos casos mais graves — o oposto do desejável.

---

## Como rodar

```bash
pip install pandas numpy matplotlib scikit-learn
# coloque enhanced_anxiety_dataset.csv em data/raw/
jupyter notebook notebooks/00_primeiro_contato.ipynb
```

No Colab não precisa instalar nada: a célula de carga abre o botão de upload.

**Fonte:** natezhang123. *Social Anxiety Dataset*. Kaggle. Licença CC0 1.0. Acesso em 26/10/2026. # modelo-machinelearning
