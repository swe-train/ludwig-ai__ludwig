import contextlib
from typing import Any, Dict, Optional, Union

from ludwig.utils import data_utils
from ludwig.utils.fs_utils import RemoteFilesystem

CredInputs = Optional[Union[str, Dict[str, Any]]]


DEFAULTS = "defaults"
ARTIFACTS = "artifacts"
DATASETS = "datasets"
CACHE = "cache"


class Credentials:
    def __init__(self, creds: Optional[Dict[str, Any]]):
        self._creds = creds

    @property
    def fs(self) -> RemoteFilesystem:
        return RemoteFilesystem(self._creds)

    @contextlib.contextmanager
    def use(self):
        with data_utils.use_credentials(self._creds):
            yield

    def to_dict(self) -> Optional[Dict[str, Any]]:
        return self._creds


class CredentialManager:
    def __init__(
        self,
        defaults: CredInputs = None,
        artifacts: CredInputs = None,
        datasets: CredInputs = None,
        cache: CredInputs = None,
    ):
        cred_inputs = {
            DEFAULTS: defaults,
            ARTIFACTS: artifacts,
            DATASETS: datasets,
            CACHE: cache,
        }

        creds = {}
        for k, v in cred_inputs.items():
            if isinstance(v, str):
                v = data_utils.load_json(v)
            creds[k] = Credentials(v)
        self.creds = cred_inputs

    @property
    def defaults(self) -> Credentials:
        return self.creds[DEFAULTS]

    @property
    def artifacts(self) -> Credentials:
        return self.creds[ARTIFACTS]

    @property
    def datasets(self) -> Credentials:
        return self.creds[DATASETS]

    @property
    def cache(self) -> Credentials:
        return self.creds[CACHE]
