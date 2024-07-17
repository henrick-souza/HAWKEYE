import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from sqlalchemy import create_engine, Column, String, Integer, MetaData, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from datetime import date

Base = declarative_base()

# Configuração do banco de dados
DATABASE_URL = "sqlite:///users.db"
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Definição da tabela de usuários
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    favorites = relationship("Favorite", back_populates="user")

class Favorite(Base):
    __tablename__ = 'favorites'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    symbol = Column(String, nullable=False)
    user = relationship("User", back_populates="favorites")

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Função para adicionar um usuário ao banco de dados
def add_user(username, password, name):
    if session.query(User).filter_by(username=username).first():
        return False
    new_user = User(username=username, password=password, name=name)
    session.add(new_user)
    session.commit()
    return True

# Função para verificar login
def check_login(username, password):
    user = session.query(User).filter_by(username=username, password=password).first()
    if user:
        st.session_state.logged_in = True
        st.session_state.user_id = user.id
        st.session_state.username = username
        st.session_state.name = user.name
    else:
        st.error("Usuário ou senha incorretos")

# Função para adicionar uma ação favorita
def add_favorite(user_id, symbol):
    if not session.query(Favorite).filter_by(user_id=user_id, symbol=symbol).first():
        new_favorite = Favorite(user_id=user_id, symbol=symbol)
        session.add(new_favorite)
        session.commit()

# Função para obter as ações favoritadas pelo usuário
def get_favorites(user_id):
    return session.query(Favorite).filter_by(user_id=user_id).all()

# Página de login
def login_page():
    st.title("Login")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Login"):
        check_login(username, password)
    if st.button("Cadastrar"):
        st.session_state.show_register = True

# Página de cadastro
def register_page():
    st.title("Cadastro")
    new_username = st.text_input("Novo usuário")
    new_password = st.text_input("Nova senha", type="password")
    new_name = st.text_input("Nome")
    if st.button("Cadastrar"):
        if add_user(new_username, new_password, new_name):
            st.success("Usuário cadastrado com sucesso! Vá para a página de login.")
            st.session_state.show_register = False
        else:
            st.error("Usuário já existe")
    if st.button("Ir para Login"):
        st.session_state.show_register = False

# Função para buscar e exibir dados da ação
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")
    info = stock.info
    return hist, info

# Função para renderizar dados da ação com plotly
def render_stock_data(ticker):
    try:
        hist, info = get_stock_data(ticker)
        if hist.empty:
            st.error("Não há dados disponíveis para esta ação.")
            return

        # Gráfico de fechamento ajustado com plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Fechamento Ajustado', line=dict(color='green')))
        fig.update_layout(
            title=f'Preço de Fechamento - {ticker}',
            xaxis_title='Data',
            yaxis_title='Preço de Fechamento Ajustado',
            template='plotly_dark'
        )
        st.plotly_chart(fig)

        # Estatísticas adicionais
        st.subheader("Estatísticas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Market Cap", value=f"{info.get('marketCap', 'N/A'):,}")
            st.metric(label="Current Price", value=f"{info.get('currentPrice', 'N/A')}")
            st.metric(label="Trailing PE", value=f"{info.get('trailingPE', 'N/A')}")
            st.metric(label="Forward PE", value=f"{info.get('forwardPE', 'N/A')}")
            st.metric(label="Revenue", value=f"{info.get('totalRevenue', 'N/A'):,}")
        with col2:
            st.metric(label="Net Income to Common", value=f"{info.get('netIncomeToCommon', 'N/A'):,}")
            st.metric(label="Debt to Equity Ratio", value=f"{info.get('debtToEquity', 'N/A')}")
            st.metric(label="Free Cashflow", value=f"{info.get('freeCashflow', 'N/A'):,}")
            st.metric(label="Dividend Yield", value=f"{info.get('dividendYield', 'N/A') * 100:.2f}%")
            st.metric(label="Return on Equity (ROE)", value=f"{info.get('returnOnEquity', 'N/A') * 100:.2f}%")
        with col3:
            st.metric(label="Volume Total", value=f"{hist['Volume'].sum():,}")
            st.metric(label="Média de Preço (Mês)", value=f"{hist['Close'].resample('M').mean().iloc[-1]:.2f}")
            st.metric(label="Média de Preço (Semana)", value=f"{hist['Close'].resample('W').mean().iloc[-1]:.2f}")
            st.metric(label="Média de Preço (Dia)", value=f"{hist['Close'].mean():.2f}")
            st.metric(label="Maior Preço do Dia", value=f"{hist['High'].max():.2f}")
            st.metric(label="Menor Preço do Dia", value=f"{hist['Low'].min():.2f}")

    except Exception as e:
        st.error(f"Erro ao obter dados da ação: {e}")

# Função para renderizar a previsão de preços
def render_price_forecast(ticker, n_days):
    try:
        DATA_INICIO = '2017-01-01'
        DATA_FIM = date.today().strftime('%Y-%m-%d')    

        df = yf.download(ticker, DATA_INICIO, DATA_FIM)
        df.reset_index(inplace=True)

        st.subheader('Tabela de valores - ' + ticker)
        st.write(df.tail(10))

        df_treino = df[['Date', 'Close']]
        df_treino = df_treino.rename(columns={"Date": 'ds', 'Close': 'y'})

        modelo = Prophet()
        modelo.fit(df_treino)

        futuro = modelo.make_future_dataframe(periods=n_days, freq='B')
        previsao = modelo.predict(futuro)

        st.subheader('Previsão')
        st.write(previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days))

        grafico1 = plot_plotly(modelo, previsao)
        st.plotly_chart(grafico1)

        grafico2 = plot_components_plotly(modelo, previsao)
        st.plotly_chart(grafico2)

    except Exception as e:
        st.error(f"Erro ao obter previsão da ação: {e}")

# Configurações da página
st.set_page_config(layout="wide")

# Inicializando o estado da sessão
if 'page' not in st.session_state:
    st.session_state.page = "Página Inicial"
if 'selected_action' not in st.session_state:
    st.session_state.selected_action = "BBAS3.SA"
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_register' not in st.session_state:
    st.session_state.show_register = False

# Função para definir a página atual
def set_page(page):
    st.session_state.page = page

# Função para definir a ação selecionada
def select_action(action):
    st.session_state.selected_action = action
    set_page("Ações")

# Estilo personalizado
st.markdown(
    """
    <style>
    :root {
        --primary-color: #000005;
        --secondary-color: #404040;
        --highlight-color: #2E2E2E;
        --text-color: white;
        --font-size-small: 0.8rem;
        --font-size-medium: 1rem;
        --font-size-large: 1.2rem;
    }
    
    body {
        font-family: 'Arial', sans-serif;
        color: var(--text-color);
    }
    
    .sidebar .sidebar-content {
        background-color: var(--primary-color);
        color: var(--text-color);
    }
    .css-1d391kg {
        background-color: var(primary-color);
    }
    .css-1aumxhk {
        background-color: var(primary-color);
    }
    .main .block-container {
        background-color: var(primary-color);
        color: var(--text-color);
    }
    .stTextInput>div>div>input {
        color: var(--text-color);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .css-1aumxhk {
        padding-top: 1rem;
    }
    .css-145kmo2 {
        display: none;
    }
    .stButton>button {
        background-color: var(--highlight-color);
        color: var(--text-color);
        border: none;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    .card {
        background-color: var(--secondary-color);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        cursor: pointer;
    }
    .card img {
        width: 50px;
        height: 50px;
        margin-right: 1rem;
    }
    .card-content {
        font-size: var(--font-size-medium);
        margin-bottom: 0.5rem;
    }
    .card-footer {
        text-align: right;
        font-size: var(--font-size-small);
    }
    .news-card {
        background-color: var(--secondary-color);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .news-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .news-header h4 {
        margin: 0;
    }
    .news-header .time {
        font-size: var(--font-size-small);
        color: #999;
    }
    .news-content {
        margin-top: 0.5rem;
        font-size: var(--font-size-small);
    }
    .news-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.5rem;
    }
    .news-footer .stars {
        color: gold;
    }
    .favorite-card {
        background-color: var(--secondary-color);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        cursor: pointer;
    }
    .favorite-card img {
        width: 50px;
        height: 50px;
        margin-right: 1rem;
    }
    .favorite-card .favorite-content {
        display: flex;
        flex-direction: column;
    }
    .favorite-card .favorite-header {
        font-size: var(--font-size-large);
        font-weight: bold;
        margin: 0;
    }
    .favorite-card .favorite-description {
        font-size: var(--font-size-medium);
        color: #aaa;
    }
    
    /* Responsividade */
    @media (max-width: 768px) {
        .stButton>button {
            font-size: 14px;
            padding: 8px;
        }
        .card-content {
            font-size: var(--font-size-small);
        }
        .favorite-card .favorite-header {
            font-size: var(--font-size-medium);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Menu lateral com a logo
st.sidebar.image("logo.png", use_column_width=True, width=150)  # Ajustar o tamanho da imagem para 150 pixels de largura
st.sidebar.title("Menu")

# Botões na aba lateral
page = st.sidebar.radio("Ir para", ["Página Inicial", "Notícias", "Ações"], index=["Página Inicial", "Notícias", "Ações"].index(st.session_state.page))

if not st.session_state.logged_in:
    if st.session_state.show_register:
        register_page()
    else:
        login_page()
else:
    if page == "Página Inicial":
        st.session_state.page = "Página Inicial"
        st.title(f"Página Inicial - Bem-vindo, {st.session_state.name}")

        col1, col2 = st.columns((2, 1))
        
        with col1:
            st.header("Lista de Ações")
            available_actions = ["BBAS3.SA", "PETR4.SA", "VALE3.SA", "AAPL", "TSLA"]
            for stock in available_actions:
                if st.button(stock, key=f"add_{stock}"):
                    add_favorite(st.session_state.user_id, stock)
            
            st.header("Favoritas")
            favorites = get_favorites(st.session_state.user_id)
            for favorite in favorites:
                st.markdown(
                    f"""
                    <div class="favorite-card">
                        <img src="https://via.placeholder.com/50?text={favorite.symbol}" alt="{favorite.symbol}">
                        <div class="favorite-content">
                            <span class="favorite-header">{favorite.symbol}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button(favorite.symbol, key=f"fav_{favorite.symbol}"):
                    select_action(favorite.symbol)

        with col2:
            st.header("Notícias")
            
            news_articles = [
                {"source": "BANCO DO BRASIL", "title": "BBAS3.SA", "summary": "O Banco do Brasil foi a primeira instituição financeira do Brasil, e atualmente é considerado um dos maiores bancos do país.", "time": "32 min ago", "stars": 3},
                {"source": "PETROBRAS", "title": "PETR4.SA", "summary": "A Petrobras é uma empresa petrolífera brasileira. Ela se dedica à exploração, produção, refino, transporte e comercialização de petróleo e seus derivados, além do gás natural.", "time": "43 min ago", "stars": 2},
                {"source": "VALE", "title": "VALE3.SA", "summary": "A Vale é uma multinacional brasileira líder na mineração e produção de minério de ferro e níquel, com operações globais e foco em sustentabilidade e inovação.", "time": "41 min ago", "stars": 2},
                {"source": "APPLE", "title": "AAPL", "summary": "A Apple é uma multinacional americana inovadora, conhecida por seus produtos eletrônicos icônicos como o iPhone, iPad e Mac, além de serviços digitais avançados.", "time": " 44 min ago", "stars": 5},
                {"source": "TESLA", "title": "TSLA", "summary": "A Tesla é uma empresa americana pioneira em veículos elétricos, energia sustentável e tecnologias de condução autônoma, com um impacto significativo no setor automotivo global.", "time": "45 min ago", "stars": 2},
            ]

            def render_stars(num_stars):
                return "⭐" * num_stars

            for article in news_articles:
                st.markdown(
                    f"""
                    <div class="news-card">
                        <div class="news-header">
                            <h4>{article['source']}</h4>
                            <span class="time">{article['time']}</span>
                        </div>
                        <div class="news-content">
                            <p><strong>{article['title']}</strong></p>
                            <p>{article['summary']}</p>
                        </div>
                        <div class="news-footer">
                            <span class="stars">{render_stars(article['stars'])}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown(
                """
                <div style="text-align: center; margin-top: 1rem;">
                    <a href="#" style="color: #fff; text-decoration: none;">See All</a>
                </div>
                """,
                unsafe_allow_html=True
            )

    elif page == "Notícias":
        st.session_state.page = "Notícias"
        st.title("Notícias")
        
        news_articles = [
            {"source": "INFOMONEY", "title": "VALE3.SA", "summary": "Vale eleva em 2,4% produção de minério de ferro no 2º trimestre", "time": "17 horas", "stars": 5},
            {"source": "FINANCE NEWS", "title": "VALE3.SA", "summary": "produção de minério de ferro alcança 80,5 milhões de toneladas no 2T24", "time": "17 horas", "stars": 5},
            {"source": "ODIA", "title": "PETR4.SA", "summary": "Aumento da Petrobras puxa alta de 1,16% na gasolina na primeira quinzena de julho", "time": "1 dia", "stars": 1},
            {"source": "SEU DINHEIRO", "title": "VALE3.SA", "summary": "Vale (VALE3) perde quase R$ 7 bilhões em valor de mercado e relatório de produção vem aí — mineradora pode ter mais de um vilão no 2T24", "time": "21 horas", "stars": 1},
            {"source": "MONEYTIMES", "title": "PETR4.SA", "summary": "Etanol: Preços saltam mais de 5% no mercado à vista após reajuste da gasolina pela Petrobras (PETR4)", "time": "1 dia", "stars": 2},
            {"source": "The New York Times", "title": "Product #B", "summary": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "time": "5 min ago", "stars": 4},
            {"source": "Reuters", "title": "Product #C", "summary": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "time": "7 min ago", "stars": 5},
            {"source": "Valor Econômico", "title": "Product #D", "summary": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "time": "9 min ago", "stars": 5},
            {"source": "forbes", "title": "Product #A", "summary": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "time": "2 min ago", "stars": 5},
            {"source": "The New York Times", "title": "Product #B", "summary": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "time": "5 min ago", "stars": 4},
            {"source": "Reuters", "title": "Product #C", "summary": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "time": "7 min ago", "stars": 5},
            {"source": "Valor Econômico", "title": "Product #D", "summary": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "time": "9 min ago", "stars": 5},
    
        ]

        def render_stars(num_stars):
            return "⭐" * num_stars

        for article in news_articles:
            st.markdown(
                f"""
                <div class="news-card">
                    <div class="news-header">
                        <h4>{article['source']}</h4>
                        <span class="time">{article['time']}</span>
                    </div>
                    <div class="news-content">
                        <p><strong>{article['title']}</strong></p>
                        <p>{article['summary']}</p>
                    </div>
                    <div class="news-footer">
                        <span class="stars">{render_stars(article['stars'])}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown(
            """
            <div style="text-align: center; margin-top: 1rem;">
                <a href="#" style="color: #fff; text-decoration: none;">See All</a>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif page == "Ações":
        st.session_state.page = "Ações"
        st.title("Ações")
        st.header(st.session_state.selected_action)
        
        # Dropdown para selecionar outra ação
        available_actions = ["BBAS3.SA", "PETR4.SA", "VALE3.SA", "AAPL", "TSLA"]
        new_action = st.selectbox("Selecione uma ação", available_actions, index=available_actions.index(st.session_state.selected_action))
        if new_action != st.session_state.selected_action:
            select_action(new_action)
            st.experimental_rerun()
        
        # Obter dados da ação
        render_stock_data(new_action)
        
        # Previsão de preços
        n_days = st.slider('Quantidade de dias de previsão', 30, 90)
        render_price_forecast(new_action, n_days)


