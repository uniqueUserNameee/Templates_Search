# Библиотека, предназначенная для работы с файловой системой
import os

# Библиотека, предназначенная для взаимодействия с интерпретатором Python
import sys

# Библиотека, предназначенная для работы с данными в формате JSON
import json

# Библиотека, предназначенная для работы с VK API
import vk_api

# Библиотека, предназначенная для работы с веб-приложениями
import requests


# Функция, которая выполняет авторизацию по логину и паролю пользователя
# Возвращает VkApiMethod(self), который позволяет обращаться к методам vk_api как к обычным классам
def vk_authorization(login, password):
    vk_session = vk_api.VkApi(login, password)
    try:
        vk_session.auth()
    except (requests.exceptions.ConnectionError, vk_api.exceptions.LoginRequired,
            vk_api.exceptions.BadPassword, vk_api.exceptions.Captcha) as exception_text:
        return exception_text
    return vk_session.get_api()


# Функция, которая записывает в JSON-файл информацию со стены пользователя или сообщества
def writing_wall_info_to_json(vk, owner_id=None, domain=None, posts_number=5, path_to_file_with_wall_info=None):
    if owner_id and not isinstance(owner_id, int):
        owner_id=None
    if not isinstance(posts_number, int) or posts_number < 1:
        posts_number = 100
    try:
        directory, _ = os.path.split(path_to_file_with_wall_info)
    except TypeError:
        path_to_file_with_wall_info = 'wall_info.json'
    else:
        if not os.path.isdir(directory):
            path_to_file_with_wall_info = 'wall_info.json'
    wall_info = vk.wall.get(owner_id=owner_id, domain=domain, count=posts_number, extended=1)
    for post_item in wall_info.get('items'):
        post_item['text'] = ' '.join(post_item.get('text').encode('cp1251', 'ignore').decode('cp1251').split())
        copy_history = post_item.get('copy_history')
        if copy_history:
            post_item['copy_history'][0]['text'] = ' '.join(copy_history[0].get('text').encode('cp1251', 'ignore').decode('cp1251').split())
    with open(path_to_file_with_wall_info, 'w', encoding='utf-8') as file_pointer:
        json.dump(wall_info, file_pointer, ensure_ascii=False, indent=2)


# Функция, которая считывает тексты из JSON- и TXT-файлов и записывает их в список
def reading_from_file(path_to_file):
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

    # Полученная со стены пользователя или сообщества информация записывается в файл в формате JSON, путь к которому был передан
    writing_wall_info_to_json(vk, domain=group_domain_name, path_to_file_with_wall_info=path_to_file_with_group_info, posts_number=100)
    
    # Считываются тексты из файла, путь к которому был передан
    texts_list = reading_from_file(path_to_file_with_group_info)
    if not texts_list:
        sys.exit()
    print(len(texts_list))
    for text in texts_list:
        print(text)