import hashlib
import importlib.util
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
# the real file's header and a row: enough to pass a column check, not the file the project was built on
LOOKALIKE = (
    b"hotel,is_canceled,lead_time,arrival_date_year,arrival_date_month,arrival_date_day_of_month,"
    b"stays_in_weekend_nights,stays_in_week_nights,adults,market_segment,adr,reservation_status\n"
    b"City Hotel,0,10,2017,July,1,0,2,2,Online TA,99.0,Check-Out\n"
)


def _load():
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def download(tmp_path, monkeypatch):
    module = _load()
    monkeypatch.setattr(module, "DEST", tmp_path / "hotels.csv")
    fetched: list[str] = []

    def fake_urlretrieve(url, dest):
        fetched.append(url)
        Path(dest).write_bytes(LOOKALIKE)

    monkeypatch.setattr(module.urllib.request, "urlretrieve", fake_urlretrieve)
    return module, fetched


def test_the_source_is_pinned_to_a_commit():
    assert re.search(r"/tidytuesday/[0-9a-f]{40}/", _load().URL)


def test_a_file_with_the_wrong_hash_is_rejected(download):
    module, fetched = download
    assert module.main() == 1
    assert len(fetched) == 1
    assert not module.DEST.exists()


def test_a_verified_file_is_kept_and_not_fetched_again(download, monkeypatch):
    module, fetched = download
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(LOOKALIKE).hexdigest())
    assert module.main() == 0 and module.DEST.read_bytes() == LOOKALIKE
    assert module.main() == 0
    assert len(fetched) == 1
