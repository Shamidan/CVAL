import os
import shutil


def clear_directory(directory_path):
    """
    Очищает содержимое указанной директории, не удаляя саму директорию.

    :param directory_path: Путь к директории для очистки.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Директория '{directory_path}' не существует.")

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Удаление файла или символической ссылки
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Удаление директории
        except Exception as e:
            raise RuntimeError(f"Ошибка при удалении '{file_path}': {e}")

    print(f"Содержимое директории '{directory_path}' очищено.")