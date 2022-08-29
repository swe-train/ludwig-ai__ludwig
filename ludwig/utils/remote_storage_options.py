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
from abc import ABCMeta
from typing import Any, Dict

from ludwig.constants import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    CLIENT_KWARGS,
    ENDPOINT_URL,
    KEY,
    S3_ENDPOINT_URL,
    SECRET,
)


class BaseRemoteStorageOptions(metaclass=ABCMeta):
    """Base Class for remote storage options to be used by fsspec guidelines for storage options structure:

    https://s3fs.readthedocs.io/en/latest/#s3-compatible-storage.
    """

    def __init__(self, key: str, secret: str, endpoint_url: str):
        self.key = os.environ.get(key, None)
        self.secret = os.environ.get(secret, None)
        self.endpoint_url = os.environ.get(endpoint_url, None)

    def get_key(self) -> str:
        return self.key

    def get_secret(self) -> str:
        return self.secret

    def get_client_kwargs(self) -> Dict[str, Any]:
        return {ENDPOINT_URL: self.endpoint_url}

    def get_storage_options(self):
        return {KEY: self.get_key(), SECRET: self.get_secret(), CLIENT_KWARGS: self.get_client_kwargs()}


class S3RemoteStorageOptions(BaseRemoteStorageOptions):
    """Get credentials from environment variables.

    May not require this class once https://github.com/aws/aws-sdk/issues/229 is done.
    """

    def __init__(self):
        super().__init__(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT_URL)
