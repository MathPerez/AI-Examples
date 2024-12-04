import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados
df = pd.read_csv('./database/Life-Expectancy-Data-Averaged.csv')

# Selecionando variáveis de entrada (features) e variável alvo (target)
X = df[['Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 'Alcohol_consumption',
        'Hepatitis_B', 'Measles', 'BMI', 'Polio', 'Diphtheria', 'Incidents_HIV',
        'GDP_per_capita', 'Population_mln', 'Thinness_ten_nineteen_years',
        'Thinness_five_nine_years', 'Schooling', 'Economy_status']]  # Features
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding para variáveis categóricas
y = df['Life_expectancy']  # Target (quantidade vendida)

# Preservar os índices originais
countries = df['Country'] 

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test, countries_train, countries_test = train_test_split(X, y, countries, test_size=0.2, random_state=42)

# Treinando o modelo de Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Predições
y_pred = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Criar DataFrame com resultados, incluindo o país
results = pd.DataFrame({'Country': countries_test, 'Actual': y_test, 'Predicted': y_pred})

# Salvar os resultados em um arquivo CSV
results.to_csv('.\\predictions_regressao_linear.csv', index=False)

print("\nResultados salvos em 'predictions_regressao_linear.csv'.")
