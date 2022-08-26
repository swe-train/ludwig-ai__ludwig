#! /usr/bin/env python
# Copyright (c) 2021 Linux Foundation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from abc import ABCMeta, abstractmethod
from typing import Any, Dict

from ludwig.constants import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, ENDPOINT_URL, MLFLOW_S3_ENDPOINT_URL


class RemoteStorageOptions(metaclass=ABCMeta):
    """Base Class for remote storage options to be used by fsspec Guidelines for storage options structure:

    https://s3fs.readthedocs.io/en/latest/#s3-compatible-storage.
    """

    @abstractmethod
    def get_key(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_secret(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_client_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError


class S3RemoteStorageOptions(RemoteStorageOptions):
    """Get credentials from environment variables."""

    def get_key(self):
        return os.environ.get(AWS_ACCESS_KEY_ID, None)

    def get_secret(self) -> str:
        return os.environ.get(AWS_SECRET_ACCESS_KEY, None)

    def get_client_kwargs(self) -> Dict[str, Any]:
        # TODO: Rename MLFLOW_S3_ENDPOINT_URL to something else
        return {
            ENDPOINT_URL: os.environ.get(MLFLOW_S3_ENDPOINT_URL, None),
        }
