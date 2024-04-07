from __future__ import print_function
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import ClassVar

base_dir = Path(os.path.dirname(__file__)).parent


class Config(BaseSettings):
    project_title: ClassVar[str] = 'Product Catalog Generator'
    debug: bool = True
    sentry_dsn: ClassVar[str] = ''
    hf_model: ClassVar[str] = 'Mohammadreza/llama-7b-lora-bslm-entity-attributes'


config = Config()
