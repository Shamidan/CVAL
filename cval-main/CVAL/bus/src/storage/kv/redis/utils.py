
async def save_to_redis(name: str, data: bytes, redis):
    # TODO: ЗАРЕФАЧИТЬ ЭТО К ХЕРАМ

    try:
        if await redis.exists(name):
            return False
        await redis.set(name, data)
    finally:
        await redis.close()
