# Facebook Prophet no Mercado Financeiro
![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<p align="center">
  <img src="https://cdn.dribbble.com/users/2543315/screenshots/16765731/media/c390676e737d8ac3f298aed5af1d37ae.png?resize=1600x1200&vertical=center" alt="META">
</p>

**Autor:** Victor Flávio P. Dornelos\
**E-mail:** victor.dornelos@hotmail.com\
**Linkedin:** [Victor Flávio Pereira Dornelos](https://www.linkedin.com/in/victor-flavio-pereira-dornelos/)

## Sumário
1. [Descrição](https://github.com/victordornelos/Serie_temporal_previdencia_social?tab=readme-ov-file#1-descrição)
2. [Objetivo](https://github.com/victordornelos/Serie_temporal_previdencia_social?tab=readme-ov-file#2-objetivo)
3. [Metodologia](https://github.com/victordornelos/Serie_temporal_previdencia_social?tab=readme-ov-file#3-metodologia)
4. [Resultados](https://github.com/victordornelos/Serie_temporal_previdencia_social?tab=readme-ov-file#4-resultados)
5. [Referências](https://github.com/victordornelos/Serie_temporal_previdencia_social?tab=readme-ov-file#5-referências)

## 1. Descrição

**Séries temporais** têm se tornado um tema amplamente debatido recentemente, impulsionado pelo avanço dos algoritmos preditivos, sendo intensamente utilizadas em Estatística, Econometria e Ciências dos Dados. Uma série temporal é simplesmente um **conjunto de observações registradas ao longo do tempo**, ordenadas cronologicamente, e geralmente com uma frequência fixa, como diário, mensal, anual ou semanal.\
Ao explorar uma série temporal, é possível analisar **tendências**, **sazonalidade** e **autocorrelação** (a dependência de uma observação em relação a outra em um período de tempo), isso pode ser extremamente útil para compreender a variável em questão. Dessa forma, séries temporais tornaram-se fundamentais para a criação de **modelos preditivos**, que visam estimar o futuro e, assim, auxiliar na tomada de decisões no presente.\
Em 2017, o **Facebook** (atualmente Meta) criou um algoritmo de código aberto chamado **Prophet**. Seu objetivo é ser uma ferramenta simples e versátil para estimar variáveis de diferentes áreas como saúde, marketing, demanda, engajamento e mercado financeiro. Neste repositório, utilizaremos o Prophet para **prever as cotações do ETF** (Exchange Traded Fund) **HASH11**, que replica o índice NCI (Nasdaq Crypto Index) voltado para criptoativos, composto principalmente por Bitcoin, Ethereum e Solana.

## 2. Objetivo
O **objetivo** deste repositório não é apenas aplicar o algoritmo Prophet do Facebook nas cotações do ETF HASH11, mas também **descrever o funcionamento do algoritmo** e realizar uma **avaliação detalhada dos resultados**. Além disso, o código e a base de dados utilizados nesse processo serão disponibilizados para consulta e reprodução dos resultados.

## 3. Algoritmo do Facebook Prophet
Como mencionado anteriormente, o algoritmo Prophet do Facebook foi projetado para ser simples e versátil, com o **objetivo de tornar-se acessível** para analistas de dados iniciantes. Modelos autorregressivos automatizados, como o AutoARIMA, têm dificuldade em lidar com a **sazonalidade de eventos humanos a longo prazo**, especialmente em situações de negócios. Como exemplos, incluem a queda no consumo de energia nos escritórios durante os feriados ou o aumento do trânsito nos dias que antecedem ou sucedessem os feriados.\
Nesse contexto, seria necessário utilizar modelos mais complexos para incluir essas situações, exigindo cientistas de dados mais experientes. Dessa forma, o Prophet foi idealizado para lidar com esses tipos de eventos, mantendo a essência da ideia de **simplicidade de uso.**\
O algoritmo Prophet trabalha com um modelo de série temporal que possui três principais componentes: tendência, sazonalidade e feriados, conforme a equação abaixo:

<div align="center">
  y(t) = g(t) + s(t) + h(t) + ϵ
</div>


Onde:
- **y(t)** são os valores previstos pelo modelo,
- **g(t)** é a função de tendência, que estima as variações não periódicas da série,
- **s(t)** representa as mudanças sazonais da série,
- **h(t)** captura o efeito dos feriados (incluindo os dias próximos a essas datas),
- **ϵ** é a parte não explicada pelo modelo.

Outro fator importante é que o algoritmo trabalha de maneira semelhante a um **modelo aditivo generalizado**, com a sazonalidade sendo aditiva,transformações são necessárias se o comportamento for multiplicativo. Para maiores detalhes sobre a parte algébrica de cada parâmetro e a automatização do modelo, consulte o artigo “[Forecasting at Scale](https://peerj.com/preprints/3190/)” de Sean J. Taylor e Benjamin Letham (o link está na seção de referências).

## 4. Metodologia
Para a realização deste projeto, será utilizada a linguagem de programação Python com o uso de Jupyter Notebooks. O primeiro passo será a coleta dos dados de cotação do HASH11, utilizando a biblioteca Yfinance. Esses dados serão armazenados em um DataFrame usando Pandas e salvos em um arquivo CSV, disponível neste repositório.

```python
# Importando bibliotecas 
import yfinance as yf
import pandas as pd

# Baixando os dados
df = pd.DataFrame()
df['HASH11'] = yf.download('HASH11.SA',start='2022-01-01')['Adj Close']

# Salvando os dados em um arquivo CSV
df.to_csv('hash11.csv')
```

Com a base de dados pronta, será realizada uma análise exploratória inicial, utilizando o Plotly para a visualização gráfica das cotações. Além disso, será feita a decomposição da série temporal utilizando o Statsmodels, gerando gráficos de tendências e de sazonalidade.

```python
# Salvando a série
date_format = lambda dates: datetime.datetime.strptime(dates, '%Y-%m-%d')

df = pd.read_csv('hash11.csv', parse_dates=['Date'], index_col='Date',
                      date_parser = date_format, usecols = ['Date', 'HASH11'])

serie = df['HASH11']
```
```python
# Gráfico das cotações
import  plotly as px
figura = px.line(title = 'Histórico de cotações de HASH11')

figura.add_scatter(x=serie.index, y=serie)


figura.update_layout(
    xaxis_title='Data',
    yaxis_title='R$',
    title_x=0.5
)

figura.show()
```
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Gerando de composição da série
decomposicao = seasonal_decompose(serie,period= len(serie) // 2)

# Separando a série
trend = decomposicao.trend
sazonal = decomposicao.seasonal
```
```python
# Gráfico de tendência
figura = px.line(title = 'Tendência das cotações de HASH11')

figura.add_scatter(x=trend.index, y=trend)

figura.update_layout(
    xaxis_title='Data',
    yaxis_title='R$',
    title_x=0.5
)

figura.show()

```
```python
# Gráfico de sazonalidade
figura = px.line(title = 'Sazonalidade de cotações do HASH11')

figura.add_scatter(x=sazonal.index, y=sazonal)

figura.update_layout(
    xaxis_title='Data',
    yaxis_title='R$',
    title_x=0.5
)

figura.show()

```
Após uma análise inicial, aplicamos o algoritmo Facebook Prophet. Vale lembrar que é necessário realizar pequenos ajustes, como a renomeação das colunas do DataFrame. Com a modelagem do FB Prophet, é possível fazer previsões das cotações. Neste teste específico, a previsão será feita para um período de 30 dias e realizado um gráfico para visualização dos dados obtidos.
```python
# Biblioteca necessária
from prophet import Prophet

# Pegando limpo os dados
df_fb = pd.read_csv('hash11.csv')

# Fazendo adequação para o modelo
df_fb = df_fb[['Date', 'HASH11']].rename(columns = {'Date': 'ds', 'HASH11': 'y'})

```
```python
# Criando o modelo
model = Prophet()
model.fit(df_fb)

# Realizando previsão com o modelo
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

```python
# Biblioteca necessária
import matplotlib.dates as mdates
import seaborn as sns

# Ajustes
df_fb['ds'] = pd.to_datetime(df_fb['ds'])
forecast['ds'] = pd.to_datetime(forecast['ds'])

# Plotando
plt.figure(figsize=(15, 6))
sns.scatterplot(x=df_fb['ds'], y=df_fb['y'], label='Dados reais', color='black')
sns.lineplot(x=forecast['ds'], y=forecast['yhat'], label='Modelo', color='blue')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.xlabel('Data')
plt.ylabel('Cotação (R$)')
plt.title('Previsão de HASH11 usando Facebook Prophet')
plt.legend()
plt.show()
```
Finalmente, após a modelagem do algoritmo, são criadas métricas para estimar os erros das previsões, com o objetivo de verificar a qualidade dos resultados obtidos.
```python
# Criando série para comparação
pred = model.make_future_dataframe(periods=0)
prev = model.predict(pred)
```
```python
# Importando bibliotecas para estimar os erros
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
```
```python
# Função para criar DF com valores dos erros
def teste_erro(serie,previsao):

    MAD = mean_absolute_error(y_true=serie,y_pred=previsao)
    MAPE = mean_absolute_percentage_error(y_true=serie,y_pred=previsao)
    MSD = mean_squared_error(y_true=serie,y_pred=previsao)

    erros = pd.DataFrame({
        'Metric': ['MAD', 'MAPE', 'MSD'],
        'Value': [MAD, MAPE, MSD]
    })
    
    return erros
```
```python
# Aplicando os testes
teste_erro(df,prev['yhat'])
```
## 5. Resultados

