import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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
X['Country'] = df['Country']  # Adicionar a coluna Country para rastrear os índices

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo de Árvore de Decisão
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train.drop(columns=['Country']), y_train)  # Remover coluna 'Country' para o treino

# Predições
y_pred = model.predict(X_test.drop(columns=['Country']))

# Recuperar a coluna 'Country' para o conjunto de teste
country_test = X_test['Country']

# Criar o DataFrame de resultados
results = pd.DataFrame({'Country': country_test, 'Actual': y_test, 'Predicted': y_pred})

# Salvar os resultados em um arquivo CSV
results.to_csv('.\\predictions_arvore_decisoes.csv', index=False)

print("\nResultados salvos em 'predictions_arvore_decisoes.csv'.")

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
