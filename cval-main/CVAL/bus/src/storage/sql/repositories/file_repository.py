import json
import uuid
from typing import Dict, List

from pydantic import BaseModel, UUID4
from sqlalchemy import delete
from src.storage.sql.models.alchemy import File
from sqlalchemy.ext.asyncio import AsyncSession
from src.storage.sql.repositories._abs_repo import BaseRepo
from sqlalchemy.future import select


class FileRepository(BaseRepo):
    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def find_by_hash(self, file_hash: str):
        result = await self.session.execute(select(File).where(File.hash == file_hash))
        return result.scalar()

    async def save_file(self, hash, json_attributes):
        existing_file = await self.find_by_hash(hash)
        if existing_file:
            return existing_file.id
        file_record = File(
            id=uuid.uuid4(),
            hash=hash,
            json_attributes=json_attributes
        )
        await self.add_object(file_record)
        return file_record.id

    async def get_all_hashes(self):
        files = await self.session.execute(select(File))
        file_models = files.scalars().all()

        for file_model in file_models:
            if isinstance(file_model.json_attributes, str):
                file_model.json_attributes = json.loads(file_model.json_attributes)

        return file_models

    async def get_hashes_by_file_ids(self, file_ids: List[str]):
        result = await self.session.execute(select(File.hash).where(File.id.in_(file_ids)))
        return result.scalars().all()

    async def get_all_files(self):
        result = await self.session.execute(select(File))
        return result.scalars().all()

    async def get_files_with_hashs(self, hashs: List[str]):
        query = select(File).where(File.hash.in_(hashs))
        result = await self.session.execute(query)
        return result.scalars().all()

    async def clear_files(self):
        await self.session.execute(delete(File))
        await self.session.commit()



class FileCreateDTO(BaseModel):
    hash: str
    json_attributes: Dict



class FileResponseDTO(BaseModel):
    id: UUID4
    hash: str
    json_attributes: Dict

    class Config:
        orm_mode = True

