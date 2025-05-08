# Bibliotecas
from requests import Session, get as requests_get
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from bs4 import BeautifulSoup
from pyrate_limiter import Duration, RequestRate, Limiter
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import re
import math

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# Configuração do yfinance para cache
yf.set_tz_cache_location("download")

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

session = CachedLimiterSession(
    limiter=Limiter(RequestRate(2, Duration.SECOND*5)),
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("yfinance.cache"),
)

# Funções Auxiliares

@st.cache_data
def carregar_carteira_ativos() -> tuple[pd.DataFrame, list, pd.DataFrame]:
    df_ion_variavel = pd.read_excel('ion-relatorio_ordens_variavel.xlsx')
    df_ion_variavel.rename(columns={'Ativo': 'CODIGO'}, inplace=True)

    df_carteira = df_ion_variavel[df_ion_variavel['Status'] == 'Confirmada'].copy()
    df_carteira['Valor'] = df_carteira['Preço'] * df_carteira['Quantidade']

    df_pendente = df_ion_variavel[df_ion_variavel['Status'] == 'Enviada'].copy()

    df_carteira['Valor Total'] = df_carteira['Valor']
    df_carteira['Valor Total Ativo'] = df_carteira.groupby('CODIGO')['Valor Total'].transform('sum')
    df_carteira['Quantidade Ativo'] = df_carteira.groupby('CODIGO')['Quantidade'].transform('sum')
    df_carteira['Preco Medio'] = (df_carteira['Valor Total Ativo'] / df_carteira['Quantidade Ativo']).round(2)

    df_carteira = df_carteira.groupby(['CODIGO', 'Tipo'], dropna=False).agg({
        'Quantidade': 'sum',
        'Valor': 'sum',
        'Preço': ['unique', 'min', 'max'],
        'Status': 'count',
        'Solicitação': ['min', 'max'],
    }).reset_index()

    lista_ativos = sorted(df_carteira['CODIGO'].unique().tolist())

    return df_carteira, lista_ativos, df_pendente

@st.cache_data
def carregar_lista_ativos_b3() -> tuple[pd.DataFrame, list]:
    df = pd.read_csv('B3.csv', sep=";", encoding='latin1')
    df.reset_index(inplace=True)
    df.columns = ['EMPRESA', 'TIPO', 'CODIGO', 'INDICES', 'OBSERVACAO']
    df['INDICES'] = df['INDICES'].str.replace('IDIV', '*IDIV').str.replace('IFIX', '*IFIX')

    indices = sorted(set(i.strip() for sublist in df['INDICES'].str.split(',') for i in sublist))
    return df, indices

@st.cache_data
def filtrar_ativos(df: pd.DataFrame, ativos: str) -> pd.DataFrame:
    ativos_list = ativos.split(',')
    pattern = "|".join(re.escape(a) for a in ativos_list)
    return df[df['CODIGO'].str.contains(pattern, case=False, na=False)]

@st.cache_data(ttl=3600)
def buscar_dados_yfinance(ativo: str):# -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    try:
        ticker = yf.Ticker(ativo + '.SA', session=session)
        #info = ticker.fast_info
        info = {}
    except Exception as e:
        st.error(f"Erro ao buscar dados no Yahoo Finance: {e}")
        st.stop()

    df_info = pd.DataFrame(info.items()).T
    df_info.columns = df_info.iloc[0]
    df_info.drop(index=0, inplace=True)

    data_final = pd.Timestamp.today()
    data_inicial = data_final - pd.DateOffset(years=10)

    historico = ticker.history(
        start=data_inicial, end=data_final, 
        actions=True, 
        auto_adjust=True, 
        back_adjust=True, 
        prepost=True)
    
    if historico.empty:
            st.warning(f"Nenhum dado histórico encontrado para o ativo {ativo}.")
            st.stop()

    return info, df_info, historico

@st.cache_data(ttl=3600)
def buscar_dados_yfinance(ativo: str, session=session):
    """
    Busca dados do ativo usando yfinance com sessão personalizada,
    incluindo dados rápidos e histórico de 10 anos.

    Usa cache por 1 hora e tratamento de erros amigável.
    """
    try:
        # Utiliza sessão com cache e controle de taxa
        ticker = yf.Tickers(ativo + ".SA", session=session)

        # Usa fast_info, que é mais leve e rápido
        info = ticker.tickers[0].fast_info or {}

        # Busca histórico de 10 anos
        historico = ticker.history(
            period="10y",
            actions=True,
            auto_adjust=True,
            back_adjust=True,
            prepost=True
        )

        if historico.empty:
            st.warning(f"Nenhum dado histórico encontrado para o ativo {ativo}.")
            st.stop()

        # Para compatibilidade com a função original, cria df_info básico
        df_info = pd.DataFrame(info.items(), columns=["Indicador", "Valor"]).T
        df_info.columns = df_info.iloc[0]
        df_info = df_info.drop(index="Indicador")

        return info, df_info, historico

#    except:
#        st.error("⚠️ Limite de requisições excedido no Yahoo Finance. Tente novamente em alguns minutos.")
#        st.stop()

    except Exception as e:
        st.error(f"Erro ao buscar dados do ativo {ativo}: {e}")
        st.stop()

@st.cache_data(ttl=3600)
def buscar_dados_investpy(ativo: str):
    """
    Busca dados históricos de ações da B3 usando investpy-reborn.
    """

    try:
        # Converte o ticker para formato aceito pela investpy (ex: 'PETR4.SA')
        ativo_formatado = ativo.upper().replace('.SA', '')

        # Busca histórico de até 10 anos (por limitação, pode usar 'dd/mm/yyyy')
        historico = investpy.stocks.get_stock_historical_data(
            stock=ativo_formatado,
            country='brazil',
            from_date='01/01/2014',
            to_date=datetime.now().strftime('%d/%m/%Y'),
            as_json=False,
            order='ascending'
        )

        historico.index = pd.to_datetime(historico.index)
        historico = historico.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })

        info = {
            'source': 'investpy',
            'symbol': ativo_formatado,
            'historico_inicio': historico.index.min(),
            'historico_fim': historico.index.max()
        }

        df_info = pd.DataFrame(info.items(), columns=['Indicador', 'Valor']).T
        df_info.columns = df_info.iloc[0]
        df_info = df_info.drop(index='Indicador')

        return info, df_info, historico

    except Exception as e:
        st.error(f"Erro ao buscar dados do ativo via investpy: {e}")
        st.stop()

@st.cache_data
def calcular_medias_moveis(df: pd.DataFrame) -> tuple[pd.DataFrame, go.Figure]:
   df['MM5d'] = df['Close'].rolling(window=5).mean()
   df['MM15d'] = df['Close'].rolling(window=15).mean()
   df['MM45d'] = df['Close'].rolling(window=45).mean()
   MMAtual = (df.iloc[-1, df.columns.get_loc('MM5d')] + df.iloc[-1, df.columns.get_loc('MM15d')] + df.iloc[-1, df.columns.get_loc('MM45d')])/3
   df['MMAtual'] = MMAtual.round(2)

   fig = go.Figure([
       go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']),
       go.Scatter(x=df.index, y=df['MM5d'], mode='lines', name='MM5d', line=dict(color='red')),
       go.Scatter(x=df.index, y=df['MM15d'], mode='lines', name='MM15d', line=dict(color='yellow')),
       go.Scatter(x=df.index, y=df['MM45d'], mode='lines', name='MM45d', line=dict(color='green')),
       go.Scatter(x=df.index, y=df['MMAtual'], mode='lines', name='MMAtual', line=dict(color='orange'))
   ])
   fig.update_layout(title='Candlestick com MM5d e MM15d', template='plotly_dark')
   return df, fig

@st.cache_data
def calcular_rsi_bollinger(df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    df['RSI'] = rsi

    df['Bollinger_MA'] = df['Close'].rolling(window=20).mean()
    df['Bollinger_Upper'] = df['Bollinger_MA'] + (df['Close'].rolling(window=20).std() * 2)
    df['Bollinger_Lower'] = df['Bollinger_MA'] - (df['Close'].rolling(window=20).std() * 2)

    fig_rsi = go.Figure([
        go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='orange'))
    ])

    fig_rsi.update_layout(title='Índice de Força Relativa (RSI)', yaxis=dict(range=[0,100]), template='plotly_dark')

    fig_bollinger = go.Figure([
        go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Preço', line=dict(color='blue')),
        go.Scatter(x=df.index, y=df['Bollinger_Upper'], mode='lines', name='Bollinger Superior', line=dict(color='red')),
        go.Scatter(x=df.index, y=df['Bollinger_Lower'], mode='lines', name='Bollinger Inferior', line=dict(color='green'))
    ])
    fig_bollinger.update_layout(title='Bandas de Bollinger', template='plotly_dark')

    return df, fig_rsi, fig_bollinger

@st.cache_data
def calcular_medias_moveis_exponenciais(df: pd.DataFrame) -> tuple[pd.DataFrame, go.Figure]:
   df['MME12d'] = df.loc[:,'Close'].ewm(span=12, adjust=False).mean()
   df['MME26d'] = df.loc[:,'Close'].ewm(span=26, adjust=False).mean()
   df['MME60d'] = df.loc[:,'Close'].ewm(span=60, adjust=False).mean()
   MMEAtual = (df['MME12d'].iloc[-1] + df['MME26d'].iloc[-1] + df['MME60d'].iloc[-1])/3
   df['MMEAtual'] = MMEAtual.round(2)
   fig = go.Figure([
      go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']),
      go.Scatter(x=df.index, y=df['MME12d'], mode='lines', name='MME12d', line=dict(color='red')),
      go.Scatter(x=df.index, y=df['MME26d'], mode='lines', name='MME26d', line=dict(color='yellow')),
      go.Scatter(x=df.index, y=df['MME60d'], mode='lines', name='MME60d', line=dict(color='green')),
      go.Scatter(x=df.index, y=df['MMEAtual'], mode='lines', name='MMEAtual', line=dict(color='orange'))
   ])
   fig.update_layout(title='Médias Móveis Exponenciais', template='plotly_dark')
   return df, fig

@st.cache_data
def calcular_macd(df: pd.DataFrame) -> go.Figure:
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=macd, mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=signal, mode='lines', name='Sinal'))
    fig.update_layout(title='MACD e Linha de Sinal', template='plotly_dark')
    return fig

@st.cache_data
def gerar_sinal_estrategia(df: pd.DataFrame) -> str:
    """
    Gera sinal automático baseado em:
    - Cruzamento de Médias Móveis Exponenciais (MME5 e MME20)
    - Cruzamento MACD e Linha de Sinal
    - Níveis do RSI
    """

    sinais = []

    # Verificar cruzamento de MME5 e MME20
    if 'MME12d' in df.columns and 'MME20d' in df.columns:
        if df['MME12d'].iloc[-2] < df['MME20d'].iloc[-2] and df['MME12d'].iloc[-1] > df['MME20d'].iloc[-1]:
            sinais.append('COMPRA')
        elif df['MME12d'].iloc[-2] > df['MME20d'].iloc[-2] and df['MME12d'].iloc[-1] < df['MME20d'].iloc[-1]:
            sinais.append('VENDA')

    # Verificar cruzamento MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    if macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
        sinais.append('COMPRA')
    elif macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
        sinais.append('VENDA')

    # Verificar RSI
    if 'RSI' in df.columns:
        ultimo_rsi = df['RSI'].dropna().iloc[-1]
        if ultimo_rsi < 30:
            sinais.append('COMPRA')
        elif ultimo_rsi > 70:
            sinais.append('VENDA')

    # Lógica Final
    if sinais.count('COMPRA') >= 2:
        return 'COMPRA'
    elif sinais.count('VENDA') >= 2:
        return 'VENDA'
    else:
        return 'MANTER'

@st.cache_data
def calcular_preco_justo_graham(info: dict) -> float | None:
    """
    Calcula o preço justo com base no modelo de Graham:
    P = √(22.5 × LPA × VPA)
    """
    try:
        lpa = info.get('trailingEps')  # lucro por ação
        vpa = info.get('bookValue')    # valor patrimonial por ação

        if lpa is not None and vpa is not None and lpa > 0 and vpa > 0:
            return round(math.sqrt(22.5 * lpa * vpa), 2)
        else:
            return None
    except:
        return None

@st.cache_data
def calcular_preco_justo_bazin(info: dict, taxa_desejada: float = 0.06) -> float | None:
    """
    Calcula o preço justo com base no modelo de Bazin:
    P = Dividendo Anual / Taxa Desejada
    """
    try:
        dividend_yield = info.get('dividendYield')  # já em proporção (ex: 0.04)
        preco_atual = info.get('currentPrice')

        if dividend_yield is not None and preco_atual is not None and dividend_yield > 0:
            dividendo_anual = preco_atual * dividend_yield
            return round(dividendo_anual / taxa_desejada, 2)/100
        else:
            return None
    except:
        return None

@st.cache_data
def exibir_distribuicao_tecnica(historico_filtrado: pd.DataFrame) -> None:
   """
   Exibe uma tabela com os principais indicadores técnicos
   e um boxplot comparando o preço atual com as médias e bandas.
   """
   ultimo = historico_filtrado.iloc[-1]
   # Boxplot
   st.subheader("📊 Distribuição das Métricas Técnicas")
   valores = {
       #'Atual': ultimo['Close'],
       'MME12d': ultimo['MME12d'],
       'MME26d': ultimo['MME26d'],
       'MME60d': ultimo['MME60d'],
       'MM15d': ultimo['MM15d'],
       'Bollinger_MA': ultimo['Bollinger_MA'],
       'Bollinger_Upper': ultimo['Bollinger_Upper'],
       'Bollinger_Lower': ultimo['Bollinger_Lower'],
   }
   df_box = pd.DataFrame.from_dict(valores, orient='index', columns=['Valor'])
   df_box['Indicador'] = df_box.index
   df_box['Preço Atual'] = ultimo['Close']

   fig_box = px.box(df_box, y='Valor', points='all', title='Boxplot: Preço Atual vs Métricas Técnicas')
   fig_box.add_scatter(
       #y=[ultimo['Close']] * len(df_box),
       y=[ultimo['Close']],
       x=df_box['Indicador'],
       mode='markers+text',
       marker=dict(color='red', size=12),
       #text=[f"R$ {ultimo['Close']:.2f}"] * len(df_box),
       text=[f"R$ {ultimo['Close']:.2f}",f"R$ {ultimo['MME12d']:.2f}"],
       textposition="top center",
       name='Preço Atual'
   )
   st.plotly_chart(fig_box, use_container_width=True)
       # Tabela
   st.subheader("📋 Tabela Técnica - Último Fechamento")
   tabela = ultimo[[
       'Close', 'RSI', 'MME12d', 'MME26d', 'MME60d',
       'MM15d', 'Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_MA'
   ]].round(2).to_frame(name='Valor')
   st.dataframe(tabela)

@st.cache_data
def consulta_agenda_dividentos(mes:str):
    # URL da agenda de dividendos de maio
    url = 'https://investidor10.com.br/acoes/dividendos/2025/maio/'

    # Requisição HTTP
    resposta = requests_get(url)
    soup = BeautifulSoup(resposta.text, 'html.parser')

    # Localiza a tabela principal
    #tabela = soup.find('table')
    tabela = soup.find_all('table')
    df = pd.read_html(str(tabela))

    # Padroniza o nome das ações (remove espaços e converte para maiúsculas)
    df['Empresa'] = df['Empresa'].str.upper().str.extract(r'([A-Z0-9]{4,5})')
    df = df.dropna(subset=['Empresa'])
    return df

    ## Filtra pelos ativos da carteira
    #df_filtrada = df[df['Empresa'].isin(carteira_acoes)].copy()
    #
    ## Renomeia e reorganiza colunas
    #df_filtrada = df_filtrada.rename(columns={
    #    'Empresa': 'Ativo',
    #    'Data Com': 'Data_Com',
    #    'Data Pagamento': 'Data_Pagamento',
    #    'Tipo': 'Tipo_Provento',
    #    'Valor': 'Valor_R$'
    #})
    #
    ## Exibe resultado
    #print("\n📆 Dividendos da CARTEIRA-AÇÕES - Maio/2025:")
    #print(df_filtrada)
    #
    ## Exporta para Excel (opcional)
    #df_filtrada.to_excel('dividendos_maio_2025.xlsx', index=False)
    #print("\nArquivo salvo: dividendos_maio_2025.xlsx")

# Inicio do App

st.title('Painel de Análise de Ações')

# Carregar dados
df_carteira, lista_ativos_carteira, df_pendente = carregar_carteira_ativos()
df_b3, lista_indices_b3 = carregar_lista_ativos_b3()

# Sidebar
st.sidebar.header('Filtros')
usar_minha_carteira = st.sidebar.checkbox('Minha Carteira', value=True)

if usar_minha_carteira:
    ativo_selecionado = st.sidebar.selectbox('Selecione um Ativo:', lista_ativos_carteira)
else:
    ativo_selecionado = st.sidebar.selectbox('Selecione um Ativo:', df_b3['CODIGO'].unique())

if ativo_selecionado:
   dados_filtrados = filtrar_ativos(df_b3, ativo_selecionado)
   dados_filtrados.set_index('EMPRESA', inplace=True)
   carteira_filtrada = filtrar_ativos(df_carteira, ativo_selecionado)
   carteira_filtrada.set_index('Tipo', inplace=True)
   info, df_info, historico = buscar_dados_yfinance(ativo_selecionado)
   st.sidebar.subheader("Intervalo de Datas Rápido")
   periodo_rapido = st.sidebar.radio("Escolha um período:", ("Personalizado", "Últimos 07 dias","Últimos 15 dias", "Últimos 30 dias", "Últimos 45 dias", "Últimos 60 dias", "Últimos 90 dias", "Últimos 6 meses", "Últimos 12 meses"))
   data_inicial = historico.index.min().to_pydatetime()
   data_final = historico.index.max().to_pydatetime()
   if periodo_rapido == "Últimos 07 dias":
       data_inicial = data_final - timedelta(days=7)
   elif periodo_rapido == "Últimos 15 dias":
       data_inicial = data_final - timedelta(days=15)
   elif periodo_rapido == "Últimos 30 dias":
       data_inicial = data_final - timedelta(days=30)
   elif periodo_rapido == "Últimos 45 dias":
       data_inicial = data_final - timedelta(days=45)
   elif periodo_rapido == "Últimos 60 dias":
       data_inicial = data_final - timedelta(days=60)
   elif periodo_rapido == "Últimos 90 dias":
       data_inicial = data_final - timedelta(days=90)
   elif periodo_rapido == "Últimos 6 meses":
       data_inicial = data_final - timedelta(days=182)
   elif periodo_rapido == "Últimos 12 meses":
       data_inicial = data_final - timedelta(days=365)
   else:
       data_inicial, data_final = st.sidebar.slider(
           'Selecione o intervalo de datas',
           min_value=historico.index.min().to_pydatetime(),
           max_value=historico.index.max().to_pydatetime(),
           value=(historico.index.min().to_pydatetime(), historico.index.max().to_pydatetime()),
           step=timedelta(days=1)
       )

   historico_filtrado = historico.loc[data_inicial:data_final]

   # Tabs para organização
   aba1, aba2, aba3 = st.tabs(["Fundamentalista", "Técnica", "Histórico"])

   with aba1:
      #st.header(f"{info.get('longName', ativo_selecionado)}")
      #st.dataframe(df_info.T, use_container_width=True)
      #st.subheader("Minha Carteira")
      #st.dataframe(carteira_filtrada.T, use_container_width=True)

      aba1_1, aba1_2, aba1_3 = st.tabs(["Sobre","Indicadores","Dividendos"])
      
      with aba1_1:
         st.subheader('Resumo da Empresa')
         st.markdown(f"""{info.get('longName')} | {info.get('website','')}""")
         st.markdown(f"{info.get('sector', 'Setor')} | {info.get('industry', 'Indústria')}")
         st.write(info.get('longBusinessSummary', 'Descrição não disponível.'))

      with aba1_2:

         preco_atual = info.get('currentPrice')
         preco_graham = calcular_preco_justo_graham(info)
         preco_bazin = calcular_preco_justo_bazin(info, taxa_desejada=0.06)

         st.subheader("Preço Justo (Modelos)")
         st.markdown(f"""[Investidor 10](https://investidor10.com.br/acoes/{str(info.get('symbol').replace('.SA',''))}/)""")

         col1, col2, col3 = st.columns(3)
         col1.metric("Preço Atual", f"R$ {preco_atual:.2f}" if preco_atual else "-")
         col2.metric("Justo (Graham)", f"R$ {preco_graham:.2f}" if preco_graham else "-")
         col3.metric("Justo (Bazin)", f"R$ {preco_bazin:.2f}" if preco_bazin else "-")

         st.subheader('Indicadores Financeiros')

         col1, col2, col3 = st.columns(3)
         col1.metric("P/L", f"{info.get('trailingPE', 0):.2f}")
         col2.metric("P/VP", f"{info.get('priceToBook', 0):.2f}")
         col3.metric("Dividend Yield", f"{info.get('dividendYield', 0):.2f}%")

         col4, col5, col6 = st.columns(3)
         col4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%")
         col5.metric("Dívida/PL", f"{info.get('debtToEquity', 0):.2f}")
         col6.metric("Liquidez Corrente", f"{info.get('currentRatio', 0):.2f}")

      with aba1_3:

         st.subheader("Indicadores de Dividendos")

         total_dividendos = historico_filtrado['Dividends'].sum()
         dividendo_mensal_medio = (historico_filtrado['Dividends'].resample('M').sum().mean())

         col1, col2 = st.columns(2)
         col1.metric("Total de Dividendos no Período", f"R$ {total_dividendos:.2f}")
         col2.metric("Média de Dividendos por Mês", f"R$ {dividendo_mensal_medio:.2f}")

         st.subheader("Próximos Dividendos")

         dividend_rate = info.get('dividendRate')
         dividend_yield = info.get('dividendYield')
         ex_dividend_date = info.get('exDividendDate')
         next_dividend_date = info.get('nextDividendDate')

         if ex_dividend_date:
             ex_dividend_date = pd.to_datetime(ex_dividend_date, unit='s')
         if next_dividend_date:
            next_dividend_date = pd.to_datetime(next_dividend_date, unit='s')

         col1, col2 = st.columns(2)

         col1.metric(
             "Dividend Rate (Anual)", 
             f"R$ {dividend_rate:.2f}" if dividend_rate else "---"
         )

         col2.metric(
             "Dividend Yield (%)", 
             f"{dividend_yield:.2f}%" if dividend_yield else "---"
         )

         col3, col4 = st.columns(2)

         col3.metric(
             "Ex-Dividend Date", 
             ex_dividend_date.strftime("%d/%m/%Y") if ex_dividend_date else "---"
         )

         col4.metric(
             "Next-Dividend Date", 
             next_dividend_date.strftime("%d/%m/%Y") if next_dividend_date else "---"
         )


   with aba2:

      st.header('Análise Técnica')

      historico_filtrado, fig = calcular_medias_moveis(historico_filtrado)
      historico_filtrado, fig_rsi, fig_bollinger = calcular_rsi_bollinger(historico_filtrado)
      historico_filtrado, fig_mme = calcular_medias_moveis_exponenciais(historico_filtrado)
      fig_macd = calcular_macd(historico_filtrado)

      tab_mm, tab_rsi, tab_bollinger, tab_mme, tab_macd = st.tabs(["Médias Móveis", "RSI", "Bandas de Bollinger", "Médias Móveis Exponenciais", "MACD"])
      
      with tab_mm:
          st.plotly_chart(fig, use_container_width=True)
      with tab_rsi:
          st.plotly_chart(fig_rsi, use_container_width=True)
      with tab_bollinger:
          st.plotly_chart(fig_bollinger, use_container_width=True)
      with tab_mme:
        st.plotly_chart(fig_mme, use_container_width=True)
        # Combinar os traços em uma nova figura
        fig_combined = go.Figure(data=fig_bollinger.data + fig_mme.data)
        fig_combined.update_layout(title='Gráficos Combinados')
        st.plotly_chart(fig_combined, use_container_width=True)
      with tab_macd:
          st.plotly_chart(fig_macd, use_container_width=True)
      
      ultimo_rsi = historico_filtrado['RSI'].dropna().iloc[-1]
      if ultimo_rsi < 30:
          st.success(f"Sinal de COMPRA detectado! (RSI = {ultimo_rsi:.2f})")
      elif ultimo_rsi > 70:
          st.error(f"Sinal de VENDA detectado! (RSI = {ultimo_rsi:.2f})")
      else:
          st.info(f"Nenhum sinal claro. (RSI = {ultimo_rsi:.2f})")
      
      st.subheader('Sinal Estratégico Automático')
      sinal = gerar_sinal_estrategia(historico_filtrado)
      
      if sinal == 'COMPRA':
          st.success('🚀 Estratégia recomenda COMPRA!')
      elif sinal == 'VENDA':
          st.error('⚠️ Estratégia recomenda VENDA!')
      else:
          st.info('ℹ️ Estratégia sugere manter posição.')

   with aba3:
      st.header('Histórico de Preços')

      #historico_filtrado_painel = historico_filtrado.copy()
      #historico_filtrado_painel['Data'] = historico_filtrado_painel.index
      #historico_filtrado_painel['Data'] = historico_filtrado_painel['Data'].dt.date
      #historico_filtrado_painel.reset_index(inplace=True, drop=True)
      #historico_filtrado_painel.set_index('Data', inplace=True)
      #historico_filtrado_painel.sort_index(ascending=False, inplace=True)
      #st.dataframe(historico_filtrado_painel[['Close','RSI','MMEAtual','MME60d','MME26d','MM15d','MME12d','Bollinger_MA','Bollinger_Upper','Bollinger_Lower']].head(1).T)
      exibir_distribuicao_tecnica(historico_filtrado)
      st.subheader("Agenda de dividendos")
      df =consula_agenda_dividentos('maio')
      st.dataframe(df)

else:
    st.warning('Nenhum ativo selecionado.')
