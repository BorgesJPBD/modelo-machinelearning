# Respira

Estimativa do nível de ansiedade a partir de hábitos de vida e sinais vitais.
Entregas 1 e 2 · Machine Learning · Projeto Final

> ⚠️ Projeto acadêmico com dados sintéticos. Não é instrumento de triagem clínica nem diagnóstico.

---

## 1. Grupo e repositório

| Campo | Preenchimento |
|---|---|
| **Projeto** | Respira |
| **Repositório** | https://github.com/BorgesJPBD/modelo-machinelearning |
| **Integrantes** | João Pedro Borges · Gustavo Olivera · Hideki · Matheus Costa |

---

## 2. Resumo da proposta

| Campo | Preenchimento |
|---|---|
| **Problema** | Dado o perfil de hábitos e sinais vitais de uma pessoa, estimar seu nível de ansiedade para que ela identifique quais hábitos mais pesam e busque ajuda antes do quadro se agravar |
| **Usuário** | a própria pessoa, num autoteste, ou um serviço de bem-estar na triagem inicial |
| **Tipo e alvo** | **regressão** · alvo: `Anxiety Level (1-10)`, escala contínua de 1 a 10 · média 3,93 |
| **Dataset** | Kaggle *Social Anxiety Dataset* · 11.000 linhas × 19 colunas · **CC0 domínio público** · zero nulos, zero duplicadas |
| **Métrica** | **MAE** em pontos da escala: diz direto "o modelo erra X ponto", que é o que o usuário entende |
| **Baseline** | prever a média: MAE 1,605 · linear só com estresse: MAE 1,221 · **modelo: MAE 0,818** |
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
| Gradient boosting | **0,818** | 1,012 | **0,779** |

*Divisão treino/teste 80/20, semente fixa. Treino: 8.800 · Teste: 2.200.*

**Resultado:** o melhor baseline é o B2, com MAE 1,221. O modelo chega a 0,818, uma **redução de 0,40 ponto** no erro.

**B1 falha de um jeito informativo.** A hipótese preguiçosa "ansiedade é a mesma coisa que estresse" fica **pior que prever a média** (R² negativo): as duas escalas vão de 1 a 10, mas têm médias diferentes, então copiar o valor erra por deslocamento. É por isso que o B2 existe — ele é o mesmo sinal, só que calibrado.

**Validação:** divisão 80/20 com semente fixa (`random_state=42`); sem estratificação, porque o alvo é contínuo. Na E2, o teste de sanidade usa validação cruzada em 5 partições no treino, e o teste continua trancado.

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
- **Causalidade reversa:** terapia e medicação aparecem associadas a ansiedade mais alta porque quem as usa já é ansioso. São consequência, não causa, e por isso não podem ser lidas como fatores de risco. Removendo as duas, o MAE sobe só de 0,818 para 0,828 — o modelo quase não depende delas.
- **O alvo é autorreferido** e correlacionado ao estresse, que também é autorreferido. Parte do que o modelo captura pode ser o jeito de a pessoa responder, não o estado dela.
- **Notas altas são raras** (8 a 10 somam ~9% dos casos), então o modelo tende a errar mais justamente nos casos mais graves — o oposto do desejável.

---

## Como rodar

```bash
# 1. ambiente virtual
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)

# 2. dependências
pip install -r requirements.txt

# 3. dados: baixa o CSV do Kaggle para data/raw/
python baixar_dados.py

# 4. notebooks (no VS Code: selecione o kernel do .venv e use "Executar Tudo")
jupyter notebook notebooks/00_primeiro_contato.ipynb
jupyter notebook notebooks/01_eda.ipynb
```

Se o download pelo `kagglehub` falhar, baixe o `.zip` em https://www.kaggle.com/datasets/natezhang123/social-anxiety-dataset e coloque `enhanced_anxiety_dataset.csv` em `data/raw/`.

No Colab não precisa instalar nada: a célula de carga abre o botão de upload.

**Fonte:** natezhang123. *Social Anxiety Dataset*. Kaggle. Licença CC0 1.0. Acesso em 26/10/2026.

---

## 7. Entrega 2 — Dados prontos

**Notebook:** `notebooks/01_eda.ipynb` (roda de cima a baixo; carga igual à da E1).

### O que a E2 entrega

| Item | Onde |
|---|---|
| EDA com as cinco perguntas, cada gráfico com Observação e Decisão | `notebooks/01_eda.ipynb`, seção 4 |
| Função `limpar(df)` (só linha a linha, antes do split) | `notebooks/01_eda.ipynb`, seção 2 |
| Split 80/20, `random_state=42` (o mesmo da E1); teste trancado | seção 3 |
| Pré-processador `ColumnTransformer` | seção 5 |
| Teste de sanidade (sem NaN, cross-val 5, comparação com baseline) | seção 6 |

### Tabela de decisões

| Coluna | Tipo | Achado na EDA | Decisão | Onde |
|---|---|---|---|---|
| Age | numérica | r = −0,07 (fraca); skew 0,10 | limite de domínio (>120 vira NaN); mediana + StandardScaler | limpar + pipeline |
| Gender | nominal (3) | 3 categorias equilibradas (2964/2934/2902) | padronizar texto; moda + one-hot, ignore | limpar + pipeline |
| Occupation | nominal (13) | 13 profissões (644 a 721 linhas); média do alvo 3,73 a 4,26 | padronizar texto; moda + one-hot, ignore | limpar + pipeline |
| Sleep Hours | numérica | r = −0,49; skew −0,24 | limite de domínio; mediana + StandardScaler | limpar + pipeline |
| Physical Activity (hrs/week) | numérica | r = −0,24; skew 0,50 | limite de domínio; mediana + StandardScaler | limpar + pipeline |
| Caffeine Intake (mg/day) | numérica | escala em centenas; r = 0,35; skew 0,32 | limite de domínio; mediana + StandardScaler | limpar + pipeline |
| Alcohol Consumption (drinks/week) | numérica | 0 é valor válido; r = 0,10; skew −0,03 | limite de domínio; mediana + StandardScaler | limpar + pipeline |
| Smoking | binária | Yes/No, ~53%/47% | padronizar para Yes/No; moda + one-hot (drop if_binary) | limpar + pipeline |
| Family History of Anxiety | binária | Yes/No, ~53%/47% | padronizar para Yes/No; moda + one-hot (drop if_binary) | limpar + pipeline |
| Stress Level (1-10) | numérica | maior correlação com o alvo (r = 0,67) | limite 1-10; mediana + StandardScaler | limpar + pipeline |
| Heart Rate (bpm) | numérica | r = 0,19; skew −0,13 | limite 30-220; mediana + StandardScaler | limpar + pipeline |
| Breathing Rate (breaths/min) | numérica | r = 0,15; skew −0,15 | limite 4-60; mediana + StandardScaler | limpar + pipeline |
| Sweating Level (1-5) | numérica (escala) | r = 0,16; escala 1-5 | limite 1-5; mediana + StandardScaler | limpar + pipeline |
| Dizziness | binária | Yes/No, ~52%/48% | padronizar para Yes/No; moda + one-hot (drop if_binary) | limpar + pipeline |
| Medication | binária | ~52% Yes; causalidade reversa (consequência da ansiedade) | mantida e sinalizada; ablação na E3 | limpar + pipeline |
| Therapy Sessions (per month) | numérica | r = 0,52; skew 1,05; causalidade reversa | mantida e sinalizada; ablação na E3; testar log1p na E3 | limpar + pipeline |
| Recent Major Life Event | binária | Yes/No, ~51%/49% | padronizar para Yes/No; moda + one-hot (drop if_binary) | limpar + pipeline |
| Diet Quality (1-10) | numérica | r = −0,22; escala 1-10 | limite 1-10; mediana + StandardScaler | limpar + pipeline |
| Anxiety Level (1-10) | alvo | contínuo; notas 8-10 raras (9,0% do treino) | não é transformado; sem stratify; atenção a notas altas na E3 | — |

### Decisões gerais

- **Sem dados faltantes, mas o pipeline imputa mesmo assim.** O usuário da API pode deixar campo vazio; `SimpleImputer` está em todos os caminhos.
- **Sem `dropna()`, sem `get_dummies`.** Tudo que aprende dos dados (imputer, scaler, encoder) está dentro do `Pipeline` e é treinado só no `X_treino`.
- **Alvo contínuo:** não há `stratify` nem `class_weight`. As notas 8 a 10 são 9,0% do treino, com um degrau após a nota 7 (1,1%, contra ~3% em cada uma de 8, 9 e 10); o tratamento dessas notas fica para a E3.
- **Nenhuma coluna vaza o alvo.** `Therapy Sessions` e `Medication` são consequência da ansiedade, não causa: ficam no modelo, sinalizadas, e a E3 repete o experimento sem elas.

### Teste de sanidade (CV 5 no treino)

Sem NaN depois do transform (8.800 linhas × 32 colunas), roda em cross-validation sem erro de categoria e fica acima do baseline: o pipeline linear reduz o MAE em 0,28 ponto frente ao B2 (1,178 → 0,901). Os baselines foram refeitos na mesma régua (CV no treino); os números da E1 foram medidos no teste.

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| B0 — média do treino | 1,569 | 2,115 | 0,000 |
| B2 — linear só com estresse | 1,178 | 1,570 | 0,448 |
| Pipeline — regressão linear | 0,901 | 1,126 | 0,716 |

O MAE do pipeline linear em CV (0,901) fica próximo ao da regressão linear completa da E1 no teste (0,892), o que indica que o pré-processador não perde informação. O gradient boosting da E1 (0,818) é a referência a bater na E3.