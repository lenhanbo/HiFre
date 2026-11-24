from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 13):
    path = Path(__file__).parents[1] / "font" / "arial.ttf"
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _display_image(tensor: torch.Tensor, width: int) -> Image.Image:
    """Render HiGAN [-1, 1] tensors as black ink on a white canvas."""
    valid = tensor.detach().float().cpu()[0, :, :max(1, int(width))].clamp(-1, 1)
    pixels = ((1.0 - (valid + 1.0) / 2.0) * 255.0).round().byte().contiguous().numpy()
    return Image.fromarray(pixels, mode="L")


def _text_width(draw: ImageDraw.ImageDraw, value: str, font) -> int:
    box = draw.textbbox((0, 0), value, font=font)
    return int(box[2] - box[0])


def save_preview(reference: torch.Tensor, reference_lengths: torch.Tensor,
                 random_style: torch.Tensor, frequency_style: torch.Tensor,
                 generated_lengths: torch.Tensor,
                 reference_texts: Sequence[str], query_texts: Sequence[str],
                 path: str | Path, max_items: int = 16,
                 writer_ids: Sequence[int | str] | None = None,
                 header: str = "", metadata: dict | None = None) -> Path:
    """Save an annotated reference/random/frequency-guided comparison panel."""
    count = min(max_items, len(reference), len(reference_texts), len(query_texts))
    if count <= 0:
        raise ValueError("Preview batch is empty")
    if writer_ids is not None and len(writer_ids) < count:
        raise ValueError("writer_ids does not cover every preview row")

    groups = [
        [_display_image(reference[index], int(reference_lengths[index]))
         for index in range(count)],
        [_display_image(random_style[index], int(generated_lengths[index]))
         for index in range(count)],
        [_display_image(frequency_style[index], int(generated_lengths[index]))
         for index in range(count)],
    ]
    titles = [
        "REAL STYLE REFERENCE",
        "RANDOM STYLE (one shared z)",
        "FREQUENCY STYLE [noise|low|original|high]",
    ]
    font = _font()
    scratch = Image.new("L", (1, 1), color=255)
    scratch_draw = ImageDraw.Draw(scratch)
    column_widths = [
        max(max(image.width for image in group), _text_width(scratch_draw, title, font))
        for group, title in zip(groups, titles)
    ]
    labels = []
    sample_metadata = []
    for index in range(count):
        writer = writer_ids[index] if writer_ids is not None else "unknown"
        reference_text = str(reference_texts[index]).replace("\n", " ")
        query_text = str(query_texts[index]).replace("\n", " ")
        labels.append(
            f"row={index:02d} | writer_index={writer} | "
            f"reference={reference_text!r} | query={query_text!r} | "
            f"widths={int(reference_lengths[index])}->{int(generated_lengths[index])}")
        sample_metadata.append({
            "row": index,
            "writer_index": writer,
            "reference_text": reference_text,
            "query_text": query_text,
            "reference_width": int(reference_lengths[index]),
            "generated_width": int(generated_lengths[index]),
        })

    margin, gap = 10, 10
    title_height, image_height, label_height = 22, 32, 21
    header_height = 24 if header else 0
    row_height = image_height + label_height + gap
    columns_width = sum(column_widths) + gap * (len(groups) - 1)
    label_width = max(_text_width(scratch_draw, label, font) for label in labels)
    header_width = _text_width(scratch_draw, header, font) if header else 0
    canvas_width = margin * 2 + max(columns_width, label_width, header_width)
    canvas_height = margin * 2 + header_height + title_height + count * row_height
    canvas = Image.new("L", (canvas_width, canvas_height), color=255)
    draw = ImageDraw.Draw(canvas)

    if header:
        draw.text((margin, margin), header, fill=0, font=font)
    title_y = margin + header_height
    x_positions = []
    current_x = margin
    for width, title in zip(column_widths, titles):
        x_positions.append(current_x)
        draw.text((current_x, title_y), title, fill=0, font=font)
        current_x += width + gap
    for row in range(count):
        image_y = title_y + title_height + row * row_height
        for column, group in enumerate(groups):
            canvas.paste(group[row], (x_positions[column], image_y))
        draw.text((margin, image_y + image_height + 2), labels[row], fill=0, font=font)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    payload = dict(metadata or {})
    payload.update({"preview": str(path), "header": header, "samples": sample_metadata})
    path.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    return path
