# -*- coding: cp1251 -*-

# Библиотека, предназначенная для работы с файловой системой
import os

# Библиотека, предназначенная для взаимодействия с интерпретатором Python
import sys

# Библиотека, предназначенная для работы с данными в формате JSON
import json

# Библиотека, предназначенная для расширенной обработки естественного языка
import spacy

# Библиотека, предназначенная для работы с VK API
import vk_api

# Библиотека, предназначенная для работы с веб-приложениями
import requests

# Библиотека, предназначенная для обработки и анализа структурированных данных
import pandas as pd


# Функция, которая выполняет авторизацию по логину и паролю пользователя
# Возвращает VkApiMethod(self), который позволяет обращаться к методам vk_api как к обычным классам
def Vk_Authorization(login, password):
    vk_session = vk_api.VkApi(login, password)
    try:
        vk_session.auth()
    except (requests.exceptions.ConnectionError, vk_api.exceptions.LoginRequired,
            vk_api.exceptions.BadPassword, vk_api.exceptions.Captcha) as exception_text:
        return exception_text
    return vk_session.get_api()


# Функция, которая записывает в JSON-файл информацию со стены пользователя или сообщества
def Writing_Wall_Info_To_Json(vk, owner_id=None, domain=None, posts_number=100, path_to_file_with_wall_info=None):
    if owner_id and not isinstance(owner_id, int):
        owner_id = None
    if not isinstance(posts_number, int) or posts_number < 1:
        posts_number = 100
    try:
        directory, filename = os.path.split(path_to_file_with_wall_info)
    except TypeError:
        path_to_file_with_wall_info = 'wall_info.json'
    else:
        if directory and not os.path.isdir(directory):
            path_to_file_with_wall_info = filename
        if os.path.splitext(filename)[1] != '.json':
            path_to_file_with_wall_info = os.path.join(directory, os.path.splitext(filename)[0] + '.json')
    wall_info = vk.wall.get(owner_id=owner_id, domain=domain, count=posts_number, extended=1)
    for post_item in wall_info.get('items'):
        post_item['text'] = ' '.join(post_item.get('text').encode('cp1251', 'ignore').decode('cp1251').split())
        copy_history = post_item.get('copy_history')
        if copy_history:
            post_item['copy_history'][0]['text'] = ' '.join(
                copy_history[0].get('text').encode('cp1251', 'ignore').decode('cp1251').split())
    with open(path_to_file_with_wall_info, 'w', encoding='utf-8') as file_pointer:
        json.dump(wall_info, file_pointer, ensure_ascii=False, indent=2)


# Функция, которая считывает тексты из JSON- и TXT-файлов и записывает их в список
def Reading_From_File(path_to_file):
    texts_list = []
    if not path_to_file or not os.path.isfile(path_to_file):
        return
    file_extension = os.path.splitext(path_to_file)[1]
    if file_extension == '.json':
        with open(path_to_file, encoding='utf-8') as file_pointer:
            try:
                wall_info = json.load(file_pointer)
            except json.JSONDecodeError:
                return
        wall_items = wall_info.get('items')
        if not isinstance(wall_items, list):
            return
        for post_item in wall_items:
            post_text = post_item.get('text')
            if post_text:
                texts_list.append(post_text)
            copy_history = post_item.get('copy_history')
            if copy_history:
                copy_history_text = copy_history[0].get('text')
                if copy_history_text:
                    texts_list.append(copy_history_text)
    elif file_extension == '.txt':
        with open(path_to_file, encoding='utf-8') as file_pointer:
            texts_list.append(file_pointer.read().replace('\n', ' '))
    return texts_list


# Функция, которая нормализует новости
def News_Normalization(news, nlp, words_number_in_news):
    tokens_list = []
    for token in nlp(news):
        word = token.lemma_
        lexeme = nlp.vocab[word]
        if not lexeme.is_stop and (word.isalpha() or word.isdigit()):
            tokens_list.append(word)
        if len(tokens_list) == words_number_in_news + 30:
            return ' '.join(tokens_list)


# Функция, внутри которой создаётся случайная топология свёрточной нейронной сети
def Building_Cnn_Model(hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])
    model.add(Conv1D(250, 5, input_shape=(None, 60000, 100), padding='valid', activation=activation_choice))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(topics_num, activation='softmax'))
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'SGD']),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':

    # Логин и пароль пользователя, необходимые для авторизации
    login = ''
    password = ''

    # Домен группы, посты которой надо распарсить
    group_domain_name = 'universe_of_f1'

    # Путь к JSON-файлу, куда будут записаны посты группы
    path_to_file_with_group_info = 'wall_info.json'

    # Путь к TXT-файлу, содержащий текст, который надо проанализировать
    path_to_file_with_group_info_txt = 'wall_info.txt'

    # Выполняется авторизация пользователя по логину и паролю
    vk = vk_authorization(login, password)
    if not isinstance(vk, vk_api.vk_api.VkApiMethod):
        sys.exit(vk)

    # Полученная со стены пользователя или сообщества информация записывается в JSON-файл, путь к которому был передан
    writing_wall_info_to_json(vk, domain=group_domain_name,
                              path_to_file_with_wall_info=path_to_file_with_group_info, posts_number=100)

    # Считываются тексты из файла, путь к которому был передан
    texts_list = reading_from_file(path_to_file_with_group_info)

    # Путь к CSV-файлу, содержащий обучающий массив текстов
    path_to_csv_file = 'lenta-ru-news.csv'

    # Список тем, новости на которые не нужны пользователю в обучающем массиве текстов
    unusable_topics_list = ['Все']

    # Минимальное количество новостей, которое должно быть на каждую тему
    min_amount_of_news_on_topic = 200

    # Количество значащих слов, которое должно быть в каждой новости
    words_number_in_news = 100

    # В столбец news_text объекта news_dataframe записываются тексты новостей, а в столбец topic - их темы
    news_dataframe = pd.read_csv(path_to_csv_file)[['title', 'text', 'tags']]
    news_dataframe['news_text'] = news_dataframe['title'].str.strip() + ' ' + news_dataframe['text'].str.strip()
    news_dataframe.rename(columns={'tags': 'topic'}, inplace=True)
    news_dataframe.drop(columns=['title', 'text'], axis=1, inplace=True)
    news_dataframe.dropna(inplace=True, subset=['news_text'])

    # Составляется список "непригодных" тем, на которые не хватает новостей для обучения нейронной сети
    topics_frequency_dict = dict(news_dataframe['topic'].value_counts())
    unusable_topics_list += [topic_item[0] for topic_item in topics_frequency_dict.items()
                             if topic_item[1] < min_amount_of_news_on_topic]

    # Удаляются новости на "непригодные" темы
    news_dataframe.set_index('topic', inplace=True)
    topics_set = set(news_dataframe.index.values)
    unusable_topics_list = set(unusable_topics_list)
    undetected_topics_list = list(unusable_topics_list - topics_set)
    news_dataframe.drop(index=list(unusable_topics_list & topics_set), inplace=True)
    topics_dict = {topic: index for index, topic in enumerate(news_dataframe.index.values)}
    news_dataframe.reset_index(inplace=True)

    # Нормализуется столбец news_text объекта news_dataframe
    nlp = spacy.load('ru_core_news_lg', exclude=['morphologizer, parser, senter, attribute_ruler, lemmatizer, ner'])
    news_dataframe['news_text'] = news_dataframe['news_text'].apply(news_normalization, args=(nlp, words_number_in_news))

    # Темы новостей заменяются на их числовые представления
    news_dataframe['topic'] = news_dataframe['topic'].map(topics_dict)

    # Тексты новостей векторизируются алгоритмом TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=0.0001)
    vectorized_news = vectorizer.fit_transform(news_dataframe['news_text'])
    vectorized_news_dataframe = pd.DataFrame(data=vectorizer.transform(news_dataframe['news_text']).todense(),
                                             columns=vectorizer.get_feature_names())
    path_to_file_with_vectorized_news = 'vectorized_news.csv'
    vectorized_news_dataframe.to_csv(path_to_file_with_vectorized_news, index=False)

    # Тексты новостей векторизируются алгоритмом векторное представление слов
    import fasttext
    import fasttext.util
    path_to_file_with_model = 'cc.ru.300.bin'
    model = fasttext.load_model(path_to_file_with_model)
    new_length_of_word_embeddings = 100
    fasttext.util.reduce_model(model, new_length_of_word_embeddings)
    vectorized_news = []
    for index, text in enumerate(news_dataframe['news_text']):
        vectorized_news.append([])
        for word in text.split():
            vectorized_news[index].append(model[word])
    vectorized_news_dataframe = pd.DataFrame(data=vectorized_news)
    path_to_file_with_vectorized_news = 'vectorized_news.csv'
    vectorized_news_dataframe.to_csv(path_to_file_with_vectorized_news, index=False)

    # Тексты новостей векторизируются алгоритмом "мешок слов"
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(news_dataframe['news_text'])
    sequences = tokenizer.texts_to_sequences(news_dataframe['news_text'])
    vectorized_news_dataframe = pd.DataFrame({'topic': news_dataframe['topic'], 'news_text': sequences})
    tokenizer_json = tokenizer.to_json()
    path_to_file_with_tokenizer = 'tokenizer.json'
    with open(path_to_file_with_tokenizer, 'w', encoding='utf-8') as file:
        file.write(json.dumps(tokenizer_json, ensure_ascii=False))
    path_to_file_with_vectorized_news = 'vectorized_news.csv'
    vectorized_news_dataframe.to_csv(path_to_file_with_vectorized_news, index=False)

    # Считываются обучающий и тестовый массивы текстов
    from tensorflow.keras import utils
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    topics_number = len(topics_list)
    train_news_dataframe = pd.read_csv('train.csv')
    x_train = pad_sequences(train_news_dataframe['news_text'].apply(json.loads).values)
    y_train = utils.to_categorical(train_news_dataframe['topic'], topics_number)
    test_news_dataframe = pd.read_csv('test.csv')
    x_test = pad_sequences(test_news_dataframe['news_text'].apply(json.loads).values)
    y_test = utils.to_categorical(test_news_dataframe['topic'], topics_number)

    # Базовая топология свёрточной нейронной сети
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
    from tensorflow.keras.callbacks import ModelCheckpoint
    number_of_words_in_dictionary = 150000
    model_cnn = Sequential()
    model_cnn.add(Embedding(number_of_words_in_dictionary, 32, input_length=words_number_in_news))
    model_cnn.add(Conv1D(250, 5, padding='valid', activation='relu'))
    model_cnn.add(GlobalMaxPooling1D())
    model_cnn.add(Dense(128, activation='relu'))
    model_cnn.add(Dense(topics_number, activation='softmax'))
    model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    path_to_file_with_best_weights = 'best_cnn_model_weights.h5'
    checkpoint_callback_cnn = ModelCheckpoint(path_to_file_with_best_weights,
                                              monitor='val_accuracy', save_best_only=True, verbose=1)
    path_to_cnn_model = 'model_cnn.json'
    model_cnn_json = model_cnn.to_json()
    with open(path_to_cnn_model, 'w') as file:
        file.write(model_cnn_json)
    model_cnn.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, callbacks=[checkpoint_callback_cnn])

    # Базовая топология сети LSTM
    from tensorflow.keras.layers import LSTM
    model_lstm = Sequential()
    model_lstm.add(Embedding(number_of_words_in_dictionary, 32, input_length=words_number_in_news))
    model_lstm.add(LSTM(16))
    model_lstm.add(Dense(topics_number, activation='softmax'))
    model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    path_to_file_with_best_weights = 'best_lstm_model_weights.h5'
    checkpoint_callback_lstm = ModelCheckpoint(path_to_file_with_best_weights, monitor='val_accuracy',
                                               save_best_only=True, verbose=1)
    path_to_lstm_model = 'model_lstm.json'
    model_lstm_json = model_lstm.to_json()
    with open(path_to_lstm_model, 'w') as file:
        file.write(model_lstm_json)
    model_lstm.fit(x_train, y_train, epochs=5, batch_size=128,
                   validation_split=0.1, callbacks=[checkpoint_callback_lstm])

    # Базовая топология сети GRU
    from tensorflow.keras.layers import GRU
    model_gru = Sequential()
    model_gru.add(Embedding(number_of_words_in_dictionary, 32, input_length=words_number_in_news))
    model_gru.add(LSTM(16))
    model_gru.add(Dense(topics_number, activation='softmax'))
    model_gru.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    path_to_file_with_best_weights = 'best_gru_model_weights.h5'
    checkpoint_callback_gru = ModelCheckpoint(path_to_file_with_best_weights, monitor='val_accuracy',
                                              save_best_only=True, verbose=1)
    path_to_gru_model = 'model_gru.json'
    model_gru_json = model_gru.to_json()
    with open(path_to_gru_model, 'w') as file:
        file.write(model_gru_json)
    model_gru.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, callbacks=[checkpoint_callback_gru])

    # Подбираются гиперпараметры свёрточной нейронной сети
    from kerastuner.tuners import RandomSearch
    tuner = RandomSearch(Building_Cnn_Model, objective='val_accuracy', max_trials=80, directory='directory_with_models')
    tuner.search_space_summary()
    tuner.search(x_train, y_train, batch_size=256, epochs=5, validation_split=0.2)