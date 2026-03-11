import uuid

from pydantic import BaseModel, UUID4
from sqlalchemy import delete
from src.storage.sql.models.alchemy import Route
from sqlalchemy.ext.asyncio import AsyncSession
from src.storage.sql.repositories._abs_repo import BaseRepo
from sqlalchemy.future import select


class RouteRepository(BaseRepo):
    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def find_by_name(self, route_name: str):
        result = await self.session.execute(select(Route).where(Route.route_name == route_name))
        return result.scalar()

    async def save_route(self, route_name: str):
        existing_route = await self.find_by_name(route_name)
        if existing_route:
            return existing_route
        route_record = Route(id=uuid.uuid4(), route_name=route_name)
        await self.add_object(route_record)
        return route_record

    async def get_all_routs(self):
        routs = await self.session.execute(select(Route))
        return routs.scalars().all()

    async def clear_routes(self):
        await self.session.execute(delete(Route))
        await self.session.commit()


class RouteCreateDTO(BaseModel):
    route_name: str


class RouteResponseDTO(BaseModel):
    id: UUID4
    route_name: str

    class Config:
        from_attributes = True
