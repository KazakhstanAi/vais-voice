from io import BytesIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from vais_voice.data.fleurs_extract import extract


def test_extract_fleurs_parquet_fixture(tmp_path):
    encoded = BytesIO()
    sf.write(encoded, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    table = pa.Table.from_pylist(
        [
            {
                "id": 17,
                "audio": {"bytes": encoded.getvalue(), "path": "17.wav"},
                "raw_transcription": "raw",
                "transcription": "normalized",
                "num_samples": 1600,
                "gender": 0,
            }
        ]
    )
    source = tmp_path / "source.parquet"
    pq.write_table(table, source)
    destination = tmp_path / "fleurs"
    assert extract(source, destination, "train", 1) == 1
    assert len(list((destination / "audio").glob("*.wav"))) == 1
    fields = (destination / "metadata.tsv").read_text(encoding="utf-8").split("\t")
    assert fields[0].startswith("train_17_")
    assert fields[2:4] == ["raw", "normalized"]
