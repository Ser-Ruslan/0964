import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


SEED = 2026
np.random.seed(SEED)

# Генерация синтетического датасета
def generate_synthetic_dataset():
    """
    Генерирует синтетический датасет для задачи классификации с 3 классами.
    
    Характеристики:
    - 400 объектов
    - 6 признаков (2 информативных, 2 избыточных, 2 неинформативных)
    - 3 класса
    - Разный масштаб признаков
    """
    
    # Базовая генерация
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=2,      # 2 информативных признака
        n_redundant=2,        # 2 избыточных признака (комбинации информативных)
        n_repeated=0,         # без повторяющихся признаков
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=SEED,
        flip_y=0.05,          # шума
        shuffle=True
    )
    
    # Создание DataFrame
    feature_names = [
        'income_level',           # информативный
        'age_group',             # информативный  
        'credit_score_derived',   # избыточный
        'debt_ratio_derived',     # избыточный
        'zip_code',              # неинформативный
        'phone_area_code'        # неинформативный
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Масштабирование признаков для создания разных масштабов
    # Признак 1 доход (большой масштаб)
    df['income_level'] = df['income_level'] * 10000 + 50000
    
    # Признак 2 возраст (средний масштаб)
    df['age_group'] = df['age_group'] * 20 + 30
    
    # Признак 3 кредитный рейтинг (малый масштаб)
    df['credit_score_derived'] = df['credit_score_derived'] * 100 + 600
    
    # Признак 4 соотношение долга (очень малый масштаб)
    df['debt_ratio_derived'] = df['debt_ratio_derived'] * 0.3 + 0.4
    
    # Признак 5 почтовый индекс (случайный, не влияет на цель)
    df['zip_code'] = np.random.randint(10000, 99999, size=len(df))
    
    # Признак 6 код города (случайный, не влияет на цель)
    df['phone_area_code'] = np.random.randint(200, 999, size=len(df))
    
    # Добавление небольшого шума к информативным признакам
    noise_scale = 0.05
    df['income_level'] += np.random.normal(0, df['income_level'].std() * noise_scale, size=len(df))
    df['age_group'] += np.random.normal(0, df['age_group'].std() * noise_scale, size=len(df))
    
    return df

# Генерация и сохранение датасета
dataset = generate_synthetic_dataset()


print("Информация о сгенерированном датасете:")
print(f"Размер: {dataset.shape}")
print(f"Классы: {dataset['target'].value_counts().sort_index().to_dict()}")

print("\nОписательная статистика признаков:")
print(dataset.describe())

# Проверка баланса классов
class_balance = dataset['target'].value_counts(normalize=True)
print(f"\nБаланс классов:")
for cls, proportion in class_balance.items():
    print(f"Класс {cls}: {proportion:.2%}")

# Корреляция с целевой переменной
correlations = dataset.drop('target', axis=1).corrwith(dataset['target'])
print(f"\nКорреляция признаков с целевой переменной:")
print(correlations.sort_values(ascending=False))


dataset.to_csv('synthetic_classification_dataset.csv', index=False)
print(f"\nДатасет сохранен в файл 'synthetic_classification_dataset.csv'")


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
dataset['target'].value_counts().plot(kind='bar')
plt.title('Распределение классов')
plt.xlabel('Класс')
plt.ylabel('Количество объектов')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
sns.boxplot(data=dataset.drop('target', axis=1))
plt.title('Распределение признаков')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('dataset_overview.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nГенерация датасета завершена!")
