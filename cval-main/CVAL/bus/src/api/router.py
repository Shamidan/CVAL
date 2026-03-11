import hashlib
import json
import os
import tempfile
from io import BytesIO
from typing import List

from fastapi import (
    Depends,
    UploadFile,
    HTTPException,
)
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.flow.compressing.zip import extract_zip
from src.integrations.depends.routable import (
    BaseRoutable,
    post,
    get,
    delete,
)
from src.integrations.depends.stub import Stub
from src.integrations.di_stubs.stubs import (
    KVStub,
    SQLSessionGetterStub,
)
from src.storage.sql.repositories import (
    FileRepository,
    AnnotationRepository,
    RouteRepository,
)

from  sqlalchemy import text

def generate_file_hash(file_data: bytes) -> str:
    return hashlib.sha256(file_data).hexdigest()


def get_metadata(file: str):
    file_name = os.path.basename(file)
    return {'metadata': {'name': str(file_name)}}


def save_zip_to_temp(file: UploadFile) -> str:
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name
    except Exception as e:
        print(f"Error occurred while saving ZIP: {e}")
    return temp_file_path


class MainView(BaseRoutable):
    # TODO убарть его
    @post('TEST_METHOD')
    async def check_ann_exist(self,
                              session: AsyncSession = Depends(Stub(SQLSessionGetterStub))):
        route_repo = RouteRepository(session)
        annotation_repo = AnnotationRepository(session)

        route = await route_repo.save_route('test')
        # sql_query = text("""
        #         SELECT annotation.id, annotation.file_id, annotation.route_id, annotation.json_data
        #         FROM annotation
        #         WHERE annotation.route_id = :route_id_1 AND annotation.json_data IS NULL
        #     """)

        sql_query = text('SELECT * FROM annotation')
        result = await session.execute(sql_query, {"route_id_1": route.id})

        annotations = [
            {"id": row.id, "file_id": row.file_id, "route_id": row.route_id, "json_data": row.json_data}
            for row in result
        ]
        return annotations



    @post('file/items')
    async def add_files(
            self,
            file: UploadFile,
            kv_storage=Depends(Stub(KVStub)),
            session: AsyncSession = Depends(Stub(SQLSessionGetterStub)),
    ) -> JSONResponse:
        """
        Метод загрузки файлов
        """

        file_repo = FileRepository(session)
        temp_zip_path = save_zip_to_temp(file)

        files, _, _ = extract_zip(temp_zip_path)

        for i, (key, file) in enumerate(files.items()):
            file_hash = generate_file_hash(file)
            file_meta = get_metadata(key)
            file_id = await file_repo.save_file(file_hash, file_meta)
            await kv_storage.save(file_hash, file)

        return JSONResponse(
            status_code=200,
            content=dict(),
        )

    @post('file/annotation/items')
    async def add_annotation(
            self,
            route_name: str,
            new_markup_zip: UploadFile,
            session: AsyncSession = Depends(Stub(SQLSessionGetterStub))
    ):
        """
        Метод загрузки аннотаций
        """

        temp_zip_path = save_zip_to_temp(new_markup_zip)
        route_repo = RouteRepository(session)
        annotation_repo = AnnotationRepository(session)
        file_repo = FileRepository(session)
        route = await route_repo.save_route(route_name)
        _, unpack_annotation, _ = extract_zip(temp_zip_path)
        # annotation = {}
        raw_annotation = json.loads(unpack_annotation.get('annotation.json').decode('utf-8'))
        files = await file_repo.get_files_with_hashs(raw_annotation.keys())

        for i in range(len(files)):
            file = files[i]
            key, val = list(raw_annotation.items())[i]
            old_ann = await annotation_repo.get_markup_by_file_id(file.id)
            if old_ann:
                print('есть такая')
                await annotation_repo.delete_annotation_by_route_id_file_id(
                    file_id=old_ann.file_id,
                    route_id=old_ann.route_id)

            file_ann = {key: val}

            await annotation_repo.save_annotation(
                file_id=file.id,
                route_id=route.id,
                json_data=file_ann,
            )

        return JSONResponse(
            status_code=200,
            content=dict(),
        )

    @post('file/annotation/items/initial')
    async def init_annotation(
            self,
            route_name: str,
            file_hashes: List[str],
            session: AsyncSession = Depends(Stub(SQLSessionGetterStub))
    ):
        """
        Метод инициализации аннотаций
        """

        route_repo = RouteRepository(session)
        annotation_repo = AnnotationRepository(session)
        file_repo = FileRepository(session)
        route = await route_repo.save_route(route_name)
        files = await file_repo.get_files_with_hashs(list(file_hashes))
        for file in files:
            file_id = file.id
            annotation = {}
            await annotation_repo.save_annotation(
                file_id=file_id,
                route_id=route.id,
                json_data=annotation,
            )
        return JSONResponse(
            status_code=200,
            content=dict(),
        )

    @get('file/{file_hash}/existence')
    async def file_existance(self, file_hash: str, kv_storage=Depends(Stub(KVStub))) -> JSONResponse:
        """
        Метод проверки наличия файла
        """

        check = await kv_storage.exists(file_hash)
        if check:
            return JSONResponse(
                status_code=200,
                content=dict(),
            )
        else:
            raise HTTPException(404)

    @get('file/items/{route_name}/annotated/items/hash')
    async def get_annotated_files_hash(
            self,
            route_name: str,
            session: AsyncSession = Depends(Stub(SQLSessionGetterStub)),
    ) -> JSONResponse:
        """
        Метод получения списка хэшей
        файлов для маршрута, у которых есть разметка
        """

        route_repo = RouteRepository(session)
        markup_repo = AnnotationRepository(session)
        file_repo = FileRepository(session)
        route = await route_repo.save_route(route_name)
        markups = await markup_repo.get_annotations_with_route_and_json_data(route.id)


        if not markups:
            raise HTTPException(status_code=404, detail="No files with markup found for this route")

        file_hashes = [markup.file.hash for markup in markups]

        return JSONResponse(
            status_code=200,
            content={'file_hashes': file_hashes},
        )

    @get('file/items/{route_name}/not-annotated/items/hash')
    async def get_unannotated_files_hash(
            self,
            route_name: str,
            session: AsyncSession = Depends(Stub(SQLSessionGetterStub)),
    ) -> JSONResponse:
        """
        Метод получения списка хэшей
        файлов для маршрута, у которых нет разметки
        """

        route_repo = RouteRepository(session)
        markup_repo = AnnotationRepository(session)
        route = await route_repo.save_route(route_name)
        markups = await markup_repo.get_annotations_with_route_and_None_json_data(route.id)

        if not markups:
            raise HTTPException(status_code=404, detail="No files with markup found for this route")

        file_hashes = [markup.file.hash for markup in markups]

        return JSONResponse(
            status_code=200,
            content={'file_hashes': file_hashes},
        )

    @get('file/{file_hash}/annotation')
    async def get_annotation_by_file_hash(
            self,
            file_hash: str,
            session: AsyncSession = Depends(Stub(SQLSessionGetterStub)),
    ) -> JSONResponse:
        """
        Метод получения разметки файла по хэшу
        """

        markup_repo = AnnotationRepository(session)
        file_repo = FileRepository(session)
        file = await file_repo.find_by_hash(file_hash)

        if not file:
            raise HTTPException(status_code=404, detail="File not found")

        markup = await markup_repo.get_markup_by_file_id(file.id)

        if markup.json_data == {}:
            raise HTTPException(status_code=404, detail="No markup found for the specified file")

        return JSONResponse(
            status_code=200,
            content={'markup': markup.json_data}
        )

    @get('file/{file_hash}/')
    async def get_file_by_hash(
            self,
            file_hash: str,
            kv_storage=Depends(Stub(KVStub)),
    ) -> StreamingResponse:
        """
        Метод получения файла по хэшу
        """

        file = await kv_storage.get_buffer(file_hash)

        if not file:
            raise HTTPException(status_code=404, detail="File not found")

        file_stream = BytesIO(file)

        return StreamingResponse(file_stream, media_type='image/jpg')

    @delete('annotation/items')
    async def delete_route_annotations(
            self,
            route_name: str,
            session: AsyncSession = Depends(Stub(SQLSessionGetterStub)),
    ) -> JSONResponse:
        """
        Метод очистки разметки для машрута
        """

        markup_repo = AnnotationRepository(session)
        route_repo = RouteRepository(session)
        route = await route_repo.save_route(route_name)
        await markup_repo.delete_by_route_id(route.id)
        return JSONResponse(
            status_code=200,
            content=dict(),
        )

    @delete('bus')
    async def clean_bus(
            self,
            kv_storage=Depends(Stub(KVStub)),
            session: AsyncSession = Depends(Stub(SQLSessionGetterStub)),
    ) -> JSONResponse:
        """
        Метод очистки файлов
        """

        route_repo = RouteRepository(session)
        file_repo = FileRepository(session)
        markup_repo = AnnotationRepository(session)

        await kv_storage.flush()

        await markup_repo.clear_markups()
        await route_repo.clear_routes()
        await file_repo.clear_files()

        return JSONResponse(
            status_code=200,
            content=dict(),
        )
