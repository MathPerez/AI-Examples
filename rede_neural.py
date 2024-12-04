import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Carregar os dados
df = pd.read_csv('./database/Life-Expectancy-Data-Averaged.csv')

# Selecionando variáveis de entrada (features) e variável alvo (target)
X = df[['Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 'Alcohol_consumption',
        'Hepatitis_B', 'Measles', 'BMI', 'Polio', 'Diphtheria', 'Incidents_HIV',
        'GDP_per_capita', 'Population_mln', 'Thinness_ten_nineteen_years',
        'Thinness_five_nine_years', 'Schooling', 'Economy_status']]  # Features
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding para variáveis categóricas
y = df['Life_expectancy']  # Target (expectativa de vida)

# Adicionando a coluna 'Country' para ser incluída nos resultados
countries = df['Country']  # Certifique-se de que 'Country' é o nome correto da coluna de país no seu dataset

# Normalizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test, countries_train, countries_test = train_test_split(
    X, y, countries, test_size=0.2, random_state=42
)

# Construção do modelo de rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compilando o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinando o modelo
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))

# Predições no conjunto de teste (usando X_test)
y_pred = model.predict(X_test)

# Converter y_pred para 1D
y_pred = y_pred.flatten()  # Ou use y_pred[:, 0] para pegar a primeira coluna

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Criar DataFrame com resultados, incluindo o país
results = pd.DataFrame({'Country': countries_test, 'Actual': y_test, 'Predicted': y_pred})

# Salvar os resultados em um arquivo CSV
results.to_csv('.\\predictions_redes_neurais.csv', index=False)

print("\nResultados salvos em 'predictions_redes_neurais.csv'.")