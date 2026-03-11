import json
import uuid
from typing import Dict
from pydantic import BaseModel, UUID4
from src.storage.sql.models.alchemy import Annotation
from sqlalchemy.ext.asyncio import AsyncSession
from src.storage.sql.repositories._abs_repo import BaseRepo
from sqlalchemy.future import select
from sqlalchemy import delete


class AnnotationRepository(BaseRepo):
    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def find_by_file_and_route(self, file_id: str, route_id: str):
        result = await self.session.execute(
            select(Annotation).where(Annotation.file_id == file_id).where(Annotation.route_id == route_id)
        )
        return result.scalar()

    async def delete_annotation_by_route_id_file_id(self, file_id, route_id):
        existing_annotation = await self.session.execute(
            select(Annotation).where(
                Annotation.file_id == file_id,
                Annotation.route_id == route_id,
            )
        )
        existing_annotation = existing_annotation.scalar()
        if existing_annotation is not None:
            await self.session.delete(existing_annotation)
            await self.session.commit()

    async def save_annotation(self, file_id, route_id, json_data):
        await self.delete_annotation_by_route_id_file_id(file_id, route_id)
        annotation_record = Annotation(
            id=uuid.uuid4(),
            file_id=file_id,
            route_id=route_id,
            json_data=json_data
        )
        await self.add_object(annotation_record)

    async def get_all_json_data(self):
        result = await self.session.execute(select(Annotation))
        markups = result.scalars().all()

        for markup in markups:
            if isinstance(markup.json_data, str):
                markup.json_data = json.loads(markup.json_data)

        return markups

    async def get_markups_by_route(self, route_id):
        result = await self.session.execute(select(Annotation).where(Annotation.route_id == route_id))
        return result.scalars().all()

    async def get_markup_by_file_id(self, file_id):
        result = await self.session.execute(select(Annotation).where(Annotation.file_id == file_id))
        return result.scalars().first()

    async def delete_by_route_id(self, route_id):
        await self.session.execute(delete(Annotation).where(Annotation.route_id == route_id))
        await self.session.commit()

    async def get_annotations_with_route_and_json_data(self, rout_id: str):
        query = (
            select(Annotation)
            .where(Annotation.route_id == rout_id)
            .where(Annotation.json_data.isnot(None))
        )
        result = await self.session.execute(query)
        annotations = result.scalars().all()
        return annotations

    async def get_annotations_with_route_and_None_json_data(self, rout_id: str):
        query = (
            select(Annotation)
            .where(Annotation.route_id == rout_id)
            .where(Annotation.json_data.is_(None))
        )
        result = await self.session.execute(query)
        annotations = result.scalars().all()
        return annotations

    async def clear_markups(self):
        await self.session.execute(delete(Annotation))
        await self.session.commit()


class MarkupCreateDTO(BaseModel):
    file_id: UUID4
    route_id: UUID4
    json_data: Dict


class MarkupResponseDTO(BaseModel):
    id: UUID4
    file_id: UUID4
    route_id: UUID4
    json_data: Dict

    class Config:
        from_attributes = True