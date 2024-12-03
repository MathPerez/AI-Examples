import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados
df = pd.read_csv('./database/data.csv', encoding='ISO-8859-1')

# Pré-processamento
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month

# Calculando o valor total da venda
df['TotalValue'] = df['Quantity'] * df['UnitPrice']

# Selecionando variáveis de entrada e variável alvo
X = df[['UnitPrice', 'Country', 'Year', 'Month']]  # Features
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding
y = df['TotalValue']  # Target (valor total da venda)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo de Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Predições
y_pred = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
