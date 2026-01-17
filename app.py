import json
import re

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.special import softmax

# Настройки страницы
st.set_page_config(
    page_title="Анализ тональности текста",
    page_icon="📈",
)

# Стили кнопок
st.markdown("""
<style>
    button {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
    }
    button:hover {
        color: #b11226 !important;
        transform: scale(1.02);
    }
    button:active {
        transform: scale(0.98);
    }
    button:focus {
        color: #b11226 !important;
    }
</style>
""", unsafe_allow_html=True)

# Стили метрик
st.markdown("""
<style>
    div[data-testid="stMetricValue"] {
        font-size: 20px !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# Стили прогресс-баров
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #b11226;
    }
</style>
""", unsafe_allow_html=True)

# Функция загрузки словаря
@st.cache_resource
def load_vocab(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab

# Функция загрузки модели
@st.cache_resource
def load_onnx_model(onnx_path):
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session

# Функция предобработки текста
def preprocess_text(text, vocab):
    text_lower = text.lower()
    text_cleaned = re.sub(r'[^\w\s]', ' ', text_lower)
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned)
    text_cleaned = re.sub(r'http\S+|www\S+|https\S+', '', text_cleaned)
    text_cleaned = text_cleaned.strip()

    indices = [vocab.get(token, vocab['<unk>']) for token in text_cleaned.split()]

    return np.array(indices, dtype=np.int64).reshape(1, -1)

# Функция для инференса
def predict_sentiment(text, session, vocab):

    input_data = preprocess_text(text, vocab)

    outputs = session.run(None, {'input': input_data})
    logits = outputs[0][0]

    probabilities = softmax(logits)

    return logits, probabilities


# Загрузка модели и словаря
VOCAB_PATH = "word2idx.json"
MODEL_PATH = "lstm_model.onnx"
vocab = load_vocab(VOCAB_PATH)
session = load_onnx_model(MODEL_PATH)
class_names = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}

def set_page(page):
    # Выбор страницы
    st.session_state.page = page

# Левое меню
with st.sidebar:
    st.button("О проекте", on_click=set_page, args=('project',))
    st.button("Данные", on_click=set_page, args=('data',))
    st.button("Метрики", on_click=set_page, args=('metrics',))
    st.button("Анализ ошибок", on_click=set_page, args=('loss',))
    st.button("Тестирование LSTM", on_click=set_page, args=('lstm',))
    st.markdown("""
    <style>
    .sidebar-footer {
        position: fixed;
        bottom: 10px;
        left: 10px;
        font-size: 0.8em;
        color: #555555;
    }
    </style>
    <div class="sidebar-footer">
        <p>© Анастасия Савелова, 2026 г.</p>
    </div>
    """, unsafe_allow_html=True)

if 'text_cleared' not in st.session_state:
    st.session_state.text_cleared = False

# Функция для очистки
def clear_text():
    st.session_state.text_area_content = ''

# Стартовая страница
if 'page' not in st.session_state:
    st.session_state.page = 'project'

# Заполнение страниц
if st.session_state.page == 'project':
    st.header("Анализ тональности текста")
    st.markdown("#### <span style='color: #b11226'>Области применения:</span>", unsafe_allow_html=True)
    st.markdown("• анализ отзывов клиентов на сайтах и маркетплейсах,\n\n"
             "• мониторинг тональности обсуждений в социальных сетях,\n\n"
             "• сортировка и приоритизация обращений в службу поддержки,\n\n"
             "• оценка результатов опросов и анкетирования,\n\n"
             "• исследование репутации бренда и медиа-анализа,\n\n"
             "• анализ тональности новостей и публикаций в СМИ.")
    st.markdown("#### <span style='color: #b11226'>Модели машинного обучения:</span>", unsafe_allow_html=True)
    st.write("• классическая модель логистической регрессии,\n\n"
             "• классическая модель наивного байеса,\n\n"
             "• рекурентная нейронная сеть LSTM,\n\n"
             "• предобученный трансформер SBERT.\n\n")

elif st.session_state.page == 'data':
    st.markdown("#### <span style='color: #b11226'>Источник данных:</span>", unsafe_allow_html=True)
    st.write("открытый датасет с kaggle:\n\nhttps://www.kaggle.com/datasets/mar1mba/russian-sentiment-dataset/data\n\n\n")
    st.markdown("#### <span style='color: #b11226'>Предобработка данных:", unsafe_allow_html=True)
    st.write("• нормализация,\n\n"
             "• очистка от шума,\n\n"
             "• лемматизация.\n\n")
    st.write("##### Распределение по классам")
    st.image('class.png', width=500)
    st.write("##### Распределение по количеству слов")
    st.write("\n\n")
    st.image('text.png', use_column_width=True,)

elif st.session_state.page == 'metrics':
    st.markdown("### <span style='color: #b11226'>Основные метрики\n\n</span>", unsafe_allow_html=True)
    colors = ['#cae1ff', '#fff4ca', '#d2ffd4', '#ffd2ca', '#e8d3ff']

    # F1-score
    data_f1 = {
        'Модель': ['Naive Bayes', 'Logistic Regression', 'LSTM + Navec', 'LSTM + Navec (small)', 'SBERT-large'],
        'Neutral': [0.60, 0.60, 0.62, 0.44, 0.59],
        'Positive': [0.78, 0.79, 0.82, 0.5, 0.79],
        'Negative': [0.68, 0.71, 0.73, 0.5, 0.74]
    }

    df_f1 = pd.DataFrame(data_f1)

    df_long = df_f1.melt(id_vars='Модель', var_name='Класс', value_name='F1-score')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df_long,
        x='Класс',
        y='F1-score',
        hue='Модель',
        ax=ax,
        palette=colors,
        width=0.7,
        linewidth=1,
        edgecolor='gray'
    )
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.7)
    plt.tight_layout()

    st.write("##### F1-score\n\n</span>", unsafe_allow_html=True)
    st.dataframe(df_f1.set_index('Модель'), use_container_width=True)
    st.pyplot(fig)

    #Accuracy + Time
    data_acc_tm = {
        'Модель': ['Naive Bayes', 'Logistic Regression', 'LSTM + Navec', 'LSTM + Navec (small)', 'SBERT-large'],
        'Accuracy': [0.68, 0.70, 0.72, 0.48, 0.71],
        'Time (sec)': [0.017, 0.037, 0.032, 0.002, 0.015],
    }
    df_acc_tm = pd.DataFrame(data_acc_tm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # График Accuracy
    sns.barplot(
        data=df_acc_tm,
        x='Модель',
        y='Accuracy',
        ax=ax1,
        palette=colors,
        width=0.5,
        linewidth=1,
        edgecolor='gray'
    )
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_ylim(0, 1)

    # График Time
    sns.barplot(
        data=df_acc_tm,
        x='Модель',
        y='Time (sec)',
        ax=ax2,
        palette=colors,
        width=0.5,
        linewidth=1,
        edgecolor='gray'
    )
    ax2.set_ylabel('Time (sec)', fontsize=12)
    ax2.set_xlabel('')
    ax2.set_xticklabels([])
    ax2.set_xticks([])

    for ax in [ax1, ax2]:
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], edgecolor='gray')
               for i in range(len(df_acc_tm['Модель']))]
    fig.legend(handles, df_acc_tm['Модель'],
               title='Модель',
               loc='center right',
               bbox_to_anchor=(0.3, 0.85),
               fontsize=11,
               title_fontsize=12)
    plt.tight_layout()

    st.write("##### Accuracy + Time\n\n</span>", unsafe_allow_html=True)
    st.dataframe(df_acc_tm.set_index('Модель'), use_container_width=True)
    st.pyplot(fig)


elif st.session_state.page == 'loss':
    st.markdown("### <span style='color: #b11226'>Матрицы ошибок\n\n</span>", unsafe_allow_html=True)
    st.write("##### Наивный байес")
    st.image('confusion_matrix_nb.png', width=500)
    st.write("##### Логистическая регрессия")
    st.image('confusion_matrix_lr.png', width=500)
    st.write("##### LSTM + NAVEC")
    st.image('confusion_matrix_lstm.png', width=500)
    st.write("##### LSTM + NAVEC (small)")
    st.image('confusion_matrix_lstm_quant.png', width=500)
    st.write("##### SBERT-large")
    st.image('confusion_matrix_sbert.png', width=500)

elif st.session_state.page == 'lstm':
    st.markdown("#### <span style='color: #b11226'>Тестирование LSTM</span>", unsafe_allow_html=True)

    text_input = st.text_area(
        "",
        height=150,
        placeholder="Введите текст для анализа тональности",
        key="text_area_content",
        value="" if st.session_state.get('text_cleared', False) else st.session_state.get('text_area_content', '')
    )

    col1, col2 = st.columns([1, 1])

    # Сброс флага очистки
    if st.session_state.text_cleared:
        st.session_state.text_cleared = False

    with col1:
        analyze_clicked = st.button("🔍 Проанализировать", use_container_width=True)

    with col2:
        st.button("🗑️ Очистить", use_container_width=True, on_click=clear_text)

    if analyze_clicked and text_input.strip():
        with st.spinner("Анализируем текст..."):
            # Предсказание
            logits, probs = predict_sentiment(text_input, session, vocab)
            predicted_class = np.argmax(probs).item()

            class_names = ["Нейтральный", "Позитивный", "Негативный"]
            class_icons = ["◯", "✔", "✘"]
            class_colors = ["#808080", "#4CAF50", "#F44336"]

            # Отображение результатов
            st.divider()
            col_result1, col_result2, col_result3 = st.columns(3)
            with col_result1:
                st.metric(
                    label="Класс",
                    value=f"{class_icons[predicted_class]} {class_names[predicted_class]}",
                )

            with col_result2:
                st.metric(
                    label="Уверенность модели",
                    value=f"{probs[predicted_class] * 100:.1f}%"
                )
            # Визуализация вероятностей
            st.write("##### Вероятности классов:\n\n")
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    st.progress(float(probs[i]), text=f"{class_names[i]}")
                    st.markdown(f"""
                    <div style='text-align: center'>
                        <h3 style='color: {class_colors[i]}'>{class_icons[i]}</h3>
                        <p><b>{probs[i] * 100:.2f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)

    if analyze_clicked and not text_input.strip():
        st.warning("Пожалуйста, введите текст для анализа")

