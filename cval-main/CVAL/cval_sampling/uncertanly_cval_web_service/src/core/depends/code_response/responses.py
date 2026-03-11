from src.core.depends.code_response.decorators import ResponseModel

RESPONSES = {
    301: 'Перенапрвлен',
    400: 'Неправильный запрос',
    401: 'Неавторизован',
    403: 'Ошибка доступа',
    404: 'Объект не найден',
    503: 'Сервер обновляется',
}

RESPONSES_SCHEMAS = {
                        k: {'model': ResponseModel, 'description': v} for k, v in RESPONSES.items()
                    } | \
                    {
                        200: {'description': 'Всё ок'}
                    }
