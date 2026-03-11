from abc import ABC
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select


class BaseRepo(ABC):

    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_object(self, object):
        self.session.add(object)
        await self.session.commit()
        await self.session.flush()

    async def find_by_id(self, model, obj_id):
        result = await self.session.execute(select(model).where(model.id == obj_id))
        return result.scalar()

    async def delete_object(self, object):
        await self.session.delete(object)
        await self.session.commit()

    async def update_object(self, object, updates: dict):
        for key, value in updates.items():
            setattr(object, key, value)
        await self.session.commit()

