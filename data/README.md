# Conjunto de Dados: Social Media Sentiments Analysis Dataset

Este diretório armazena o dataset utilizado para extração de atributos léxicos via Processamento de Linguagem Natural (PLN) e cálculo do Índice de Temperatura via Lógica Nebulosa (Fuzzy).

## Fonte do Dataset
* **Dataset Oficial no Kaggle:** [Social Media Sentiments Analysis Dataset](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset?resource=download)
* **Autor(a):** Kashish Parmar
* **Arquivo:** `sentimentdataset.csv`

## Estrutura do Arquivo (`sentimentdataset.csv`)
O conjunto de dados contém registros de postagens coletadas em redes sociais (Twitter, Instagram, Facebook, etc.) com atributos textuais, métricas de engajamento e anotações categóricas:

| Coluna | Tipo | Descrição | Uso no Modelo |
|---|---|---|---|
| `Text` | Texto | Conteúdo textual da publicação | Entrada do extrator léxico TextBlob (extrai $P$ e $S$) |
| `Sentiment` | Categórico | Emoção/sentimento original anotado (ex: Positive, Negative, Joy, Anger, etc.) | Utilizado para validação cruzada / análise de boxplot |
| `Timestamp` | Data/Hora | Momento da postagem | Análise temporal |
| `User` | Texto | Identificador do usuário | Metadado |
| `Platform` | Categórico | Rede social de origem (Twitter, Instagram, Facebook, etc.) | Filtragem e segmentação |
| `Hashtags` | Texto | Tópicos e hashtags indexadas | Contexto temático |
| `Retweets` | Numérico | Quantidade de compartilhamentos/retweets | Componente do Engajamento ($E$) |
| `Likes` | Numérico | Quantidade de curtidas/reações | Componente do Engajamento ($E$) |
| `Country` | Texto | País de origem do post | Metadado geográfico |
| `Year`, `Month`, `Day`, `Hour` | Numérico | Decomposição temporal da publicação | Metadado temporal |

## Observações de Pré-processamento
1. **Espaços em Branco:** Os campos de texto e categorias contêm espaços em branco residuais no início e no fim das strings, exigindo limpeza via `.str.strip()`.
2. **Escala de Engajamento:** Para mitigar a distribuição em cauda longa de interações de redes sociais, as colunas `Likes` e `Retweets` passam por atenuação logarítmica $\log(1 + \text{Likes} + \text{Retweets})$ antes da normalização linear para o intervalo $[0, 1]$.
