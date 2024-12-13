import os
import pandas as pd
import json
import matplotlib.pyplot as plt

# Задание 1. Загрузка набора данных
file_path = 'transactions_data.csv'  # Путь к файлу
file_size = os.path.getsize(file_path) / (1024 * 1024)  # Размер файла в МБ
print(f"Размер файла на диске: {file_size:.2f} МБ")

# Чтение данных чанками (оптимизация для больших файлов)
chunk_size = 10000
selected_columns = ['date', 'amount', 'client_id', 'use_chip', 'merchant_state', 'card_id', 'merchant_id', 'mcc', 'zip', 'errors']
chunks = pd.read_csv(file_path, usecols=selected_columns, chunksize=chunk_size)

# Объединение всех чанков
data_chunks = []
for chunk in chunks:
    chunk['amount'] = chunk['amount'].str.replace('$', '').astype(float)
    chunk['date'] = pd.to_datetime(chunk['date'], format='%Y-%m-%d %H:%M:%S')  # Формат даты
    chunk['zip'] = pd.to_numeric(chunk['zip'], errors='coerce')  # Преобразование zip в числовой формат
    data_chunks.append(chunk)
raw_data = pd.concat(data_chunks, ignore_index=True)

print("Пример данных:")
print(raw_data.head())

# Задание 2. Анализ набора данных
print("\n== Задание 2: Анализ набора данных ==")
memory_usage_total = raw_data.memory_usage(deep=True).sum() / (1024 * 1024)
print(f"Объем памяти при загрузке: {memory_usage_total:.2f} МБ")

# Сохранение статистики
column_stats = []
for col in raw_data.columns:
    memory_usage = raw_data[col].memory_usage(deep=True) / (1024 * 1024)
    memory_share = memory_usage / memory_usage_total
    data_type = raw_data[col].dtype
    column_stats.append({
        'column': col,
        'memory_usage_mb': memory_usage,
        'memory_share': memory_share,
        'data_type': str(data_type)
    })

column_stats_sorted = sorted(column_stats, key=lambda x: x['memory_usage_mb'], reverse=True)
os.makedirs('./results', exist_ok=True)
with open('results/data_statistics_no_optimization.json', 'w', encoding='utf-8') as f:
    json.dump(column_stats_sorted, f, ensure_ascii=False, indent=4)

# Оптимизация данных
for col in raw_data.select_dtypes(include=['object']).columns:
    unique_values = raw_data[col].nunique()
    total_values = len(raw_data[col])
    if unique_values / total_values < 0.5:
        raw_data[col] = raw_data[col].astype('category')

for col in raw_data.select_dtypes(include=['int']).columns:
    raw_data[col] = pd.to_numeric(raw_data[col], downcast='integer')

for col in raw_data.select_dtypes(include=['float']).columns:
    raw_data[col] = pd.to_numeric(raw_data[col], downcast='float')

# Повторный анализ
memory_usage_total_optimized = raw_data.memory_usage(deep=True).sum() / (1024 * 1024)
print(f"Объем памяти после оптимизации: {memory_usage_total_optimized:.2f} МБ")

# Сохранение оптимизированной статистики
column_stats_optimized = []
for col in raw_data.columns:
    memory_usage = raw_data[col].memory_usage(deep=True) / (1024 * 1024)
    memory_share = memory_usage / memory_usage_total_optimized
    data_type = raw_data[col].dtype
    column_stats_optimized.append({
        'column': col,
        'memory_usage_mb': memory_usage,
        'memory_share': memory_share,
        'data_type': str(data_type)
    })

column_stats_optimized_sorted = sorted(column_stats_optimized, key=lambda x: x['memory_usage_mb'], reverse=True)
with open('results/data_statistics_optimized.json', 'w', encoding='utf-8') as f:
    json.dump(column_stats_optimized_sorted, f, ensure_ascii=False, indent=4)

print("Оптимизированная статистика сохранена в data_statistics_optimized.json")

# Задание 8. Сохранение фильтрованного набора
print("\n== Задание 8: Сохранение фильтрованного набора ==")
filtered_data = raw_data[selected_columns]
filtered_data.to_csv('results/filtered_data.csv', index=False)
print("Фильтрованные данные сохранены в results/filtered_data.csv")

# Задание 9. Построение графиков
print("\n== Задание 9: Построение графиков ==")

# 1. Линейный график
plt.figure()
time_series = filtered_data.groupby('date')['amount'].sum()
time_series.plot(kind='line', title='Сумма транзакций по времени', ylabel='Сумма транзакций')
plt.savefig('results/line_chart.png')
plt.close()

# 2. Столбчатый график
plt.figure()
filtered_data['use_chip'].value_counts().plot(kind='bar', title='Частота использования чипа', xlabel='Тип транзакции', ylabel='Частота')
plt.savefig('results/bar_chart.png')
plt.close()

# 3. Круговая диаграмма
plt.figure()
filtered_data['merchant_state'].value_counts().head(5).plot(kind='pie', title='Частота по штатам', autopct='%1.1f%%')
plt.ylabel('')
plt.savefig('results/pie_chart.png')
plt.close()

# 4. Scatter plot
plt.figure()
filtered_data.plot(kind='scatter', x='client_id', y='amount', title='Сумма транзакций против ID клиента')
plt.savefig('results/scatter_plot.png')
plt.close()

# 5. Гистограмма
plt.figure()
filtered_data['amount'].plot(kind='hist', bins=30, title='Распределение сумм транзакций')
plt.xlabel('Сумма транзакций')
plt.ylabel('Частота')
plt.savefig('results/histogram.png')
plt.close()
