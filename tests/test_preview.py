import json

from PIL import Image
import torch

from higan_frequency.preview import save_preview


def test_annotated_preview_and_metadata_are_written(tmp_path):
    images = torch.full((2, 1, 32, 48), -1.0)
    images[:, :, 8:24, 10:30] = 1.0
    output = tmp_path / "step_00000001.png"
    save_preview(
        images, torch.tensor([32, 48]), images, images,
        torch.tensor([48, 32]), ["reference-a", "reference-b"],
        ["query-a", "query-b"], output, writer_ids=[7, 9],
        header="TRAIN | epoch=001 | global_step=00000001",
        metadata={"seed": 123})
    with Image.open(output) as preview:
        assert preview.width > 3 * 32
        assert preview.height > 2 * 32
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 123
    assert metadata["samples"][0] == {
        "row": 0,
        "writer_index": 7,
        "reference_text": "reference-a",
        "query_text": "query-a",
        "reference_width": 32,
        "generated_width": 48,
    }
