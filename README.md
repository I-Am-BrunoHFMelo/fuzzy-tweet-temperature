# Sistema de Inferência Fuzzy para Quantificação da Temperatura de Postagens em Redes Sociais utilizando PNL e Engajamento

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/I-Am-BrunoHFMelo/fuzzy-tweet-temperature/blob/main/notebooks/pipeline_fuzzy.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Fuzzy: scikit-fuzzy](https://img.shields.io/badge/Fuzzy-scikit--fuzzy-orange.svg)](https://pythonhosted.org/scikit-fuzzy/)
[![Sentiment: VADER](https://img.shields.io/badge/Sentiment-VADER%20(NLTK)-blueviolet.svg)](https://www.nltk.org/_modules/nltk/sentiment/vader.html)
[![Subjectivity: TextBlob](https://img.shields.io/badge/Subjectivity-TextBlob-green.svg)](https://textblob.readthedocs.io/)

> **Resumo:** Este projeto propõe e implementa um Sistema de Inferência Fuzzy (FIS) do tipo Mamdani para quantificar a **temperatura** (nível de intensidade, comoção, potencial de debate e relevância afetiva) de publicações em redes sociais. O modelo correlaciona atributos textuais contínuos de Processamento de Linguagem Natural — **polaridade via VADER** (otimizado para microblogs, emojis e pontuações enfáticas) e **subjetividade via TextBlob** — com métricas de topologia de rede (engajamento atenuado por logaritmo), superando as limitações de limiares rígidos (*crisp thresholds*) de abordagens tradicionais baseadas em `if-else`.

---

## 📌 Sumário
1. [Motivação e Contribuição Teórica](#-motivação-e-contribuição-teórica)
2. [Dataset](#-dataset)
3. [Modelagem Matemática do Sistema Fuzzy](#-modelagem-matemática-do-sistema-fuzzy)
   - [Variáveis de Entrada (Antecedentes)](#variáveis-de-entrada-antecedentes)
   - [Variável de Saída (Consequente)](#variável-de-saída-consequente)
4. [Base de Regras de Inferência (Mamdani - 27 Regras)](#-base-de-regras-de-inferência-mamdani---27-regras)
5. [Estrutura do Repositório](#-estrutura-do-repositório)
6. [Instalação e Execução](#-instalação-e-execução)
   - [No Google Colab](#opção-1-google-colab-recomendado)
   - [Ambiente Local](#opção-2-execução-local)
7. [Resultados Esperados e Validação](#-resultados-esperados-e-validação)
8. [Licença e Autor](#-licença-e-autor)

---

## 🎯 Motivação e Contribuição Teórica

Classificadores baseados em regras rígidas (*crisp*) enfrentam dificuldades severas para analisar o discurso em microblogs:
1. **Eliminação de Limiares Arbitrários:** Em um sistema binário, uma frase com polaridade $0.49$ pode ser rotulada como "Neutra", enquanto $0.51$ vira "Positiva". A Lógica Nebulosa permite que um valor pertença simultaneamente a múltiplos conjuntos com diferentes graus de pertinência, eliminando descontinuidades.
2. **Superfície de Decisão Não-Linear:** A relevância de uma postagem não é aditiva pura. Um texto passional e negativo com zero engajamento representa apenas um desabafo isolado (temperatura morna); já o mesmo texto com alto engajamento reflete um boicote ou comoção crítica (temperatura quente). O FIS modela essa intersecção não-linear de forma suave.
3. **Absorção de Incerteza Léxica:** Dicionários de sentimento possuem imprecisões inerentes diante de sarcasmo, gírias e pontuação emotiva. O uso do **VADER** captura as especificidades das redes sociais (emojis como `😡` e `💪`, pontuações como `!!!` e termos em caixa alta), e o motor difuso absorve incertezas residuais na etapa de agregação e defuzzificação.

---

## 📊 Dataset

O projeto utiliza o [Social Media Sentiments Analysis Dataset](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset?resource=download) (disponível no Kaggle por Kashish Parmar).

* **Arquivo:** `data/sentimentdataset.csv`
* **Colunas utilizadas:**
  * `Text`: Conteúdo textual bruto submetido à extração de polaridade (VADER) e subjetividade (TextBlob).
  * `Sentiment`: Rótulo de emoção/sentimento categórico original (*Joy, Anger, Surprise, etc.*), utilizado na validação cruzada.
  * `Likes` e `Retweets`: Métricas de propagação combinadas e normalizadas.

> **Tratamento de Cauda Pesada (Power Law):** Métricas de curtidas e retweets em redes sociais concentram-se em poucos posts virais. Para evitar que a maioria das publicações colapsasse para $E \approx 0$, aplicou-se compressão logarítmica antes da escala Min-Max:
> $$E_{\text{raw}} = \log(1 + \text{Likes} + \text{Retweets})$$
> $$E = \frac{E_{\text{raw}} - \\min(E_{\text{raw}})}{\\max(E_{\text{raw}}) - \\min(E_{\text{raw}})} \in [0, 1]$$

---

## 📐 Modelagem Matemática do Sistema Fuzzy

Todas as funções de pertinência foram formuladas para satisfazer a **partição da unidade** ($\sum \mu_i(x) = 1, \forall x$), garantindo coerência métrica e ausência de descontinuidades na inferência.

### Variáveis de Entrada (Antecedentes)

#### 1. Polaridade ($P \in [-1, 1]$)
Extraída via **VADER** (`SentimentIntensityAnalyzer.polarity_scores['compound']`). Mapeia a valência do texto calibrada para redes sociais:

$$\mu_{\text{Negativa}}(P) = \begin{cases} -P, & -1 \le P \le 0 \\ 0, & P > 0 \end{cases}$$

$$\mu_{\text{Neutra}}(P) = \begin{cases} P + 1, & -1 \le P \le 0 \\ 1 - P, & 0 < P \le 1 \end{cases}$$

$$\mu_{\text{Positiva}}(P) = \begin{cases} 0, & P < 0 \\ P, & 0 \le P \le 1 \end{cases}$$

#### 2. Subjetividade ($S \in [0, 1]$)
Extraída via **TextBlob** (`TextBlob.sentiment.subjectivity`). Mapeia o teor opinativo/emocional versus factual:

$$\mu_{\text{Factual}}(S) = \begin{cases} 1 - 2S, & 0 \le S \le 0.5 \\ 0, & S > 0.5 \end{cases}$$

$$\mu_{\text{Mista}}(S) = \begin{cases} 2S, & 0 \le S \le 0.5 \\ 2 - 2S, & 0.5 < S \le 1 \end{cases}$$

$$\mu_{\text{Passional}}(S) = \begin{cases} 0, & S < 0.5 \\ 2S - 1, & 0.5 \le S \le 1 \end{cases}$$

#### 3. Engajamento Normalizado ($E \in [0, 1]$)
Distribuição simétrica análoga à subjetividade, medindo alcance e propagação:

$$\mu_{\text{Baixo}}(E) = \begin{cases} 1 - 2E, & 0 \le E \le 0.5 \\ 0, & E > 0.5 \end{cases}$$

$$\mu_{\text{Médio}}(E) = \begin{cases} 2E, & 0 \le E \le 0.5 \\ 2 - 2E, & 0.5 < E \le 1 \end{cases}$$

$$\mu_{\text{Alto}}(E) = \begin{cases} 0, & E < 0.5 \\ 2E - 1, & 0.5 \le E \le 1 \end{cases}$$

---

### Variável de Saída (Consequente)

#### Índice de Temperatura ($T \in [0, 100]$)
Quantifica a intensidade do debate e a temperatura da postagem:

$$\mu_{\text{Fria}}(T) = \begin{cases} 1 - \frac{T}{50}, & 0 \le T \le 50 \\ 0, & T > 50 \end{cases}$$

$$\mu_{\text{Morna}}(T) = \begin{cases} \frac{T}{50}, & 0 \le T \le 50 \\ 2 - \frac{T}{50}, & 50 < T \le 100 \end{cases}$$

$$\mu_{\text{Quente}}(T) = \begin{cases} 0, & T < 50 \\ \frac{T}{50} - 1, & 50 \le T \le 100 \end{cases}$$

---

## 📋 Base de Regras de Inferência (Mamdani - 27 Regras)

O cruzamento exaustivo de 3 antecedentes com 3 conjuntos linguísticos gera $3 \times 3 \times 3 = 27$ regras de inferência Mamdani. O método de defuzzificação empregado é o **Centróide** (*Center of Gravity - COG*).

| # | Polaridade ($P$) | Subjetividade ($S$) | Engajamento ($E$) | Temperatura ($T$) | Racional Semântico / Justificativa |
|---|---|---|---|---|---|
| **R01** | Negativa | Factual | Baixo | **Fria** | Relato objetivo de problema sem repercussão na rede |
| **R02** | Negativa | Mista | Baixo | **Fria** | Reclamação branda sem tração |
| **R03** | Negativa | Passional | Baixo | **Morna** | Desabafo raivoso isolado (descontentamento pontual) |
| **R04** | Neutra | Factual | Baixo | **Fria** | Informação puramente factual de baixo alcance |
| **R05** | Neutra | Mista | Baixo | **Fria** | Opinião branda e neutra |
| **R06** | Neutra | Passional | Baixo | **Fria** | Publicação ambígua sem engajamento |
| **R07** | Positiva | Factual | Baixo | **Fria** | Elogio ou relato positivo rotineiro |
| **R08** | Positiva | Mista | Baixo | **Fria** | Postagem positiva comum |
| **R09** | Positiva | Passional | Baixo | **Morna** | Elogio eufórico individual de alcance restrito |
| **R10** | Negativa | Factual | Médio | **Morna** | Falha técnica ou problema objetivo ganhando atenção |
| **R11** | Negativa | Mista | Médio | **Morna** | Crítica em estágio de tração intermediária |
| **R12** | Negativa | Passional | Médio | **Quente** | Reclamação inflamada acumulando discussões |
| **R13** | Neutra | Factual | Médio | **Fria** | Notícia/fato informativo com circulação normal |
| **R14** | Neutra | Mista | Médio | **Morna** | Discussão de tema neutro com engajamento |
| **R15** | Neutra | Passional | Médio | **Morna** | Conteúdo com ironia/sarcasmo sob tração |
| **R16** | Positiva | Factual | Médio | **Fria** | Divulgação de resultado positivo |
| **R17** | Positiva | Mista | Médio | **Morna** | Engajamento favorável crescente |
| **R18** | Positiva | Passional | Médio | **Quente** | Celebração coletiva / hype moderado |
| **R19** | Negativa | Factual | Alto | **Morna** | Incidente grave amplamente noticiado (tom sóbrio) |
| **R20** | Negativa | Mista | Alto | **Quente** | Onda viral de críticas |
| **R21** | Negativa | Passional | Alto | **Quente** | Boicote agudo, cancelamento ou indignação massiva |
| **R22** | Neutra | Factual | Alto | **Morna** | Comunicado oficial de alcance massivo |
| **R23** | Neutra | Mista | Alto | **Morna** | Debate viral sem polarização extrema |
| **R24** | Neutra | Passional | Alto | **Quente** | Meme altamente viral ou polarização implícita |
| **R25** | Positiva | Factual | Alto | **Morna** | Fato positivo altamente compartilhado |
| **R26** | Positiva | Mista | Alto | **Quente** | Campanha ou anúncio com ampla aclamação |
| **R27** | Positiva | Passional | Alto | **Quente** | Viralização eufórica máxima / comoção de fandom |

---

## 📁 Estrutura do Repositório

```text
fuzzy-tweet-temperature/
├── LICENSE                        # Licença MIT
├── README.md                      # Documentação acadêmica e formalização teórica
├── requirements.txt               # Dependências do projeto
├── data/
│   ├── README.md                  # Dicionário e fonte dos dados
│   └── sentimentdataset.csv       # Dataset do Kaggle
├── notebooks/
│   └── pipeline_fuzzy.ipynb       # Pipeline completo: VADER + TextBlob, Fuzzy, 3D e Análises
└── figures/
    └── .gitkeep                   # Destino dos gráficos gerados em alta resolução
```

---

## 🚀 Instalação e Execução

### Opção 1: Google Colab (Recomendado)
Clique no badge abaixo para abrir diretamente o notebook no Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/I-Am-BrunoHFMelo/fuzzy-tweet-temperature/blob/main/notebooks/pipeline_fuzzy.ipynb)

No Colab, a primeira célula instalará as dependências (`scikit-fuzzy`, `textblob`, `nltk`) e executará o pipeline sem necessidade de configuração prévia.

### Opção 2: Execução Local
1. Clone o repositório:
```bash
git clone https://github.com/I-Am-BrunoHFMelo/fuzzy-tweet-temperature.git
cd fuzzy-tweet-temperature
```

2. Crie um ambiente virtual e instale as dependências:
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

3. Inicie o Jupyter Lab / Notebook:
```bash
jupyter notebook notebooks/pipeline_fuzzy.ipynb
```

---

## 📈 Resultados e Validação

O notebook gera automaticamente os seguintes artefatos para composição do artigo acadêmico:
1. **Gráficos das Funções de Pertinência ($\mu$):** Visualização das partições triangulares e trapezoidais.
2. **Superfícies de Controle 3D (*Control Surface*):**
   * Polaridade (VADER) $\times$ Engajamento $\rightarrow$ Temperatura
   * Subjetividade (TextBlob) $\times$ Engajamento $\rightarrow$ Temperatura
3. **Distribuição por Emoção Original (Validação Cruzada):** Boxplot demonstrando a coerência entre o índice contínuo de temperatura e categorias semânticas (*Anger, Joy, Surprise, Neutral*).
4. **Estudo de Casos Extremos:** Tabela com os posts de maior e menor temperatura no dataset.

---

## 📄 Licença e Autor

* **Autor:** Bruno Henrique Freitas de Melo
* **Licença:** [MIT License](LICENSE)
