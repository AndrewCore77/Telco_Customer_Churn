# Прогнозирование Оттока Клиентов в Телекоме

End-to-end ML case study на датасете IBM Telco Customer Churn.  
Этот проект собран как portfolio-grade пример того, как пройти путь от сырых клиентских данных до продуктово-ориентированной retention-стратегии: не просто предсказать churn, а перевести выводы модели в конкретные действия для маркетинга, CRM и продуктовой команды.

## Бизнес-проблема

Телеком-компании теряют регулярную выручку, когда клиенты отключают сервис, понижают тариф или перестают пользоваться платными услугами. Бизнес-цель проекта — заранее находить клиентов с повышенным риском оттока и успевать вмешаться до того, как выручка будет потеряна.

В этом проекте churn prediction рассматривается как задача принятия решения:

- кто с высокой вероятностью уйдет;
- насколько уверена модель;
- каких клиентов стоит таргетировать в первую очередь;
- какой trade-off возникает между поимкой большего числа churn-клиентов и перерасходом retention-бюджета.

## Почему это важно

Прогноз оттока ценен только тогда, когда он помогает бизнесу распределять ограниченные retention-ресурсы лучше, чем массовая одинаковая кампания на всю базу.

На практике это означает:

- снижать предотвратимые потери выручки из-за пропущенных churn-клиентов;
- приоритизировать high-risk клиентов для CRM-команды;
- выбирать threshold с учетом стоимости кампаний и доступной емкости каналов;
- понимать, какие клиентские паттерны сильнее всего связаны с оттоком;
- превращать сигналы модели в продуктовые и маркетинговые гипотезы.

Проект специально выстроен так, чтобы показать: ML-работа может быть одновременно технически качественной и коммерчески полезной.

## Бизнес-эффект

Финальная модель здесь подается не как «просто классификатор», а как инструмент для retention-решений.

Ключевые бизнес-выводы:

- Выбранная модель, **Logistic Regression**, показала лучший общий баланс для proactive retention с `ROC-AUC = 0.842` и `Recall = 0.791` на дефолтном threshold.
- Threshold tuning показывает, что модель поддерживает разные operating modes:
  - **Aggressive retention (`0.30`)** ловит около `92.5%` churn-клиентов, но создает большую нагрузку на CRM.
  - **Balanced (`0.55`)** ловит около `75.4%` churn-клиентов и заметно сокращает число false positives.
  - **Conservative (`0.70`)** уменьшает объем кампании, но пропускает значительно больше реальных churn-клиентов.
- Если использовать `MonthlyCharges` как простой proxy выручки, balanced threshold выглядит разумным компромиссом между расходом retention-бюджета и риском потерянной выручки.

Важно: в датасете нет реальной стоимости кампаний, contribution margin и фактического uplift, поэтому финансовая интерпретация в проекте носит **концептуальный**, а не строго финансовый характер.

## Датасет

**Источник:** IBM Telco Customer Churn  
**Наблюдений:** 7,043 клиента  
**Исходных колонок:** 21  
**Target:** `Churn`

Датасет включает:

- демографию клиентов,
- срок жизни аккаунта,
- подключенные телеком-услуги,
- тип контракта,
- поведение в оплате,
- ежемесячные и накопленные платежи.

Замечания по качеству данных:

- `TotalCharges` в raw-файле хранится как текст и содержит пустые строки для части клиентов с нулевым tenure;
- target умеренно несбалансирован: примерно `73.5%` non-churn против `26.5%` churn.

## Структура проекта

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       └── telco_churn_clean.csv
├── notebooks/
│   └── churn_prediction.ipynb
└── src/
    ├── data_preprocessing.py
    ├── evaluate.py
    ├── features.py
    ├── train.py
    └── utils.py
```

## Методология

Workflow намеренно организован как полноценный ML-проект, а не как быстрый ноутбук-эксперимент.

### 1. Подготовка данных

- загружен raw IBM Telco dataset;
- провалидирована схема и проверены дубликаты;
- обработаны скрытые пропуски в `TotalCharges`;
- нормализованы бизнес-читабельные категориальные признаки;
- сохранен clean dataset для дальнейшей работы.

### 2. Exploratory Data Analysis

- проанализировано распределение churn и class imbalance;
- сравнены churn vs non-churn по ключевым числовым признакам;
- профилированы категориальные драйверы churn, такие как contract type, internet service, payment method и add-on services;
- сформулированы предмодельные бизнес-гипотезы для retention.

### 3. Feature Engineering

Добавлены легкие и интерпретируемые признаки, которые улучшают моделирование без data leakage:

- `tenure_group`
- `is_new_customer`
- `num_services`
- `avg_monthly_spend_proxy`
- `has_auto_payment`

### 4. Preprocessing

- выполнен stratified train/test split;
- разделены numeric и categorical feature groups;
- применен one-hot encoding для категориальных признаков;
- добавлен scaling для Logistic Regression;
- весь preprocessing строится без leakage и fit-ится только на train data.

### 5. Моделирование

- baseline classifier;
- Logistic Regression;
- Random Forest.

### 6. Evaluation и decision layer

- accuracy, precision, recall, ROC-AUC;
- confusion matrices;
- сравнение ROC curves;
- threshold tuning;
- risk segmentation;
- бизнес-интерпретация false positives и false negatives.

## Использованные модели

### Baseline

Dummy classifier, который нужен, чтобы показать, почему одной accuracy недостаточно в churn-задаче с несбалансированным target.

### Logistic Regression

Выбрана как основная модель, потому что сочетает:

- хорошее качество ранжирования;
- высокий recall для proactive retention;
- прозрачную интерпретацию через коэффициенты;
- более простую коммуникацию с продуктовой и маркетинговой командами.

### Random Forest

Используется как нелинейный benchmark, чтобы улавливать interaction effects и сравнивать precision/recall trade-offs с линейной моделью.

## Метрики

Основные метрики проекта:

- **Accuracy**
- **Precision**
- **Recall**
- **ROC-AUC**

### Результаты на test set

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline | 0.7346 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.7324 | 0.4975 | 0.7914 | 0.8420 |
| Random Forest | 0.7729 | 0.5551 | 0.7273 | 0.8405 |

Интерпретация:

- **Baseline** выглядит приемлемо по accuracy, но полностью проваливается в ловле churn-клиентов;
- **Random Forest** сильнее по accuracy и precision;
- **Logistic Regression** сильнее по recall и немного лучше по ROC-AUC, поэтому лучше подходит для proactive retention.

## Ключевые инсайты

Проект показывает консистентную бизнес-историю, которая повторяется в EDA, коэффициентах Logistic Regression и feature importance у Random Forest.

### Самые рискованные клиенты не случайны

High-risk клиенты чаще всего:

- находятся на **month-to-month contracts**;
- имеют **короткий tenure**, особенно в первый год;
- пользуются **fiber optic internet**;
- платят через **electronic check**;
- не используют **Online Security**;
- не используют **Tech Support**.

### Churn выглядит как сочетание ценового давления, слабой привязки и низкой продуктовой вовлеченности

Сильнейшие churn-сигналы говорят о том, что клиенты уходят не из-за одного фактора, а из-за комбинации нескольких условий:

- низкий switching friction;
- нестабильность на раннем жизненном цикле;
- высокие или особенно заметные ежемесячные платежи;
- слабое использование “sticky” сервисных фич.

### Первый год особенно важен

Период onboarding и early-life — наиболее опасное окно по риску оттока. Это значит, что retention должен начинаться раньше, чем клиент явно проявит недовольство.

### Самый сильный high-risk сегмент реально пригоден для бизнеса

Один из самых сильных сегментов в проекте:

`Month-to-month + tenure <= 12 months + fiber optic`

На test sample этот сегмент показал особенно высокий фактический churn и выглядит сильным кандидатом для targeted CRM treatment.

## Бизнес-рекомендации

1. **В первую очередь таргетировать first-year month-to-month fiber customers.** Это самый очевидный high-risk кластер и лучший кандидат для приоритетного вмешательства.
2. **Тестировать contract-migration offers для рискованных month-to-month клиентов.** Ограниченные по времени annual-plan incentives или bundle credits могут работать эффективнее, чем широкие скидки для всех.
3. **Предлагать bundle или trial для Online Security и Tech Support у хрупких internet-клиентов.** Эти услуги стабильно связаны с более низким churn и могут усиливать stickiness.
4. **Запустить payment-friction кампанию для клиентов с electronic check.** Стимулировать переход на autopay через удобство, снижение friction и недорогие incentives.
5. **Использовать score-based retention tiers вместо одной общей кампании.** High-risk клиенты должны получать самые дорогие и сильные вмешательства, medium-risk — более легкие nudges.
6. **Построить early-life retention journey.** Сфокусироваться на первых 90-180 днях: onboarding, first-bill communication, раннее выявление проблем.
7. **Тестировать retention actions по сегментам, а не только глобально.** Правильный save playbook для price-sensitive newcomer не будет тем же самым, что и для billing-friction клиента.

## Ограничения

Это сильный portfolio case study, но его все равно нужно читать с правильными оговорками:

- датасет **observational**, поэтому связи не стоит трактовать как доказанную причинность;
- данные представляют собой **single snapshot**, а не полную историю клиентских событий;
- нет явных данных о **campaign cost**, **margin**, **LTV** или **uplift**;
- часть признаков коррелирована между собой, особенно tenure-related и service-related features;
- анализ threshold и потерь выручки использует `MonthlyCharges` как **концептуальный proxy**, а не как точную финансовую модель.

## Следующие шаги

Проект можно логично усилить в нескольких направлениях:

- добавить probability calibration и PR-curve analysis;
- оценить churn impact через assumptions по margin или CLV;
- ввести segment-specific thresholds для разных retention-каналов;
- отслеживать post-intervention outcomes и двигаться в сторону uplift modeling;
- упаковать pipeline в более формальный training/inference workflow;
- добавить experiment tracking и model monitoring.

## Почему этот проект сильный

Для recruiter или hiring manager этот проект показывает больше, чем базовое умение пользоваться scikit-learn:

- **Product-oriented framing:** работа начинается с бизнес-проблемы, а не с выбора модели;
- **Reproducible ML workflow:** data preparation, feature engineering, preprocessing, training и evaluation вынесены в `src/`;
- **Decision-aware evaluation:** анализ выходит за пределы одной accuracy и включает threshold tuning, campaign trade-offs и operating policy;
- **Interpretability with actionability:** выводы модели переведены в конкретные клиентские сегменты и retention-гипотезы;
- **Production thinking:** в проекте описано, как скоринги можно встроить в CRM workflow и как их мониторить со временем.

## Как запустить локально

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/churn_prediction.ipynb
```

Если вы хотите использовать reusable Python modules напрямую, основные артефакты проекта такие:

- notebook: `notebooks/churn_prediction.ipynb`
- preprocessing helpers: `src/data_preprocessing.py`
- feature engineering и preprocessing: `src/features.py`
- training logic: `src/train.py`
- evaluation и interpretation helpers: `src/evaluate.py`

## Итог

Этот проект показывает, как превратить стандартный churn dataset в более сильный portfolio case study:

- с чистым ML pipeline;
- с понятным сравнением моделей;
- с интерпретируемыми факторами churn;
- с threshold tuning, привязанным к бизнес trade-offs;
- и с конкретной retention-стратегией, которую реально могут использовать product, CRM и marketing команды.
