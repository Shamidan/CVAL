import uuid

from sqlalchemy import ForeignKey, JSON, Text, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.storage.sql.core.base_model import OrmBase


class File(OrmBase):
    __tablename__ = "file"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    hash: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    json_attributes: Mapped[dict] = mapped_column(JSON, )


class Route(OrmBase):
    __tablename__ = "route"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    route_name: Mapped[str] = mapped_column(String(256), nullable=False)


class Annotation(OrmBase):
    __tablename__ = "annotation"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    file_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("file.id"))
    route_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("route.id"))
    json_data: Mapped[dict] = mapped_column(JSON, nullable=True)
    file = relationship('File', uselist=False, lazy="selectin",)
