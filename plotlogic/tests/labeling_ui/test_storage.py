import os
from labeling_ui.lib.storage import save_panel_labels


def test_save_panel_labels(tmp_path):
    labels_dir = tmp_path / "labels" / "panels"
    labels_dir.mkdir(parents=True)
    save_panel_labels(str(labels_dir), "paper1", [{"panel_id": "p1"}])
    assert (labels_dir / "paper1.json").exists()


def test_save_panel_labels_merges_existing(tmp_path):
    labels_dir = tmp_path / "labels" / "panels"
    labels_dir.mkdir(parents=True)
    existing = {
        "panels": [
            {"panel_id": "p1", "panel_bbox_xyxy": [0, 0, 1, 1]},
            {"panel_id": "p2", "panel_bbox_xyxy": [1, 1, 2, 2]},
        ]
    }
    (labels_dir / "paper1.json").write_text(
        __import__("json").dumps(existing), encoding="utf-8"
    )

    save_panel_labels(
        str(labels_dir),
        "paper1",
        [{"panel_id": "p2", "panel_bbox_xyxy": [9, 9, 10, 10]}],
    )

    data = __import__("json").loads((labels_dir / "paper1.json").read_text("utf-8"))
    by_id = {item["panel_id"]: item for item in data["panels"]}
    assert by_id["p1"]["panel_bbox_xyxy"] == [0, 0, 1, 1]
    assert by_id["p2"]["panel_bbox_xyxy"] == [9, 9, 10, 10]


def test_build_hierarchy_from_files(tmp_path):
    root = tmp_path / "dataset"
    (root / "figures" / "0001").mkdir(parents=True)
    (root / "panels" / "0001").mkdir(parents=True)
    (root / "links").mkdir(parents=True)

    (root / "figures" / "0001" / "0001_F02__src.png").write_bytes(b"x")
    (root / "panels" / "0001" / "0001_F02-P01_chart.png").write_bytes(b"x")

    (root / "links" / "component_index.json").write_text(
        '{\"C0001\":{\"panel_id\":\"0001_F02-P01_chart\",\"panel_path\":\"panels/0001/0001_F02-P01_chart.png\",\"component_type\":\"plot_frame\",\"bbox_xyxy\":[1,2,3,4],\"crop_path\":\"components/0001/C0001.png\",\"overlay_path\":\"overlays/0001/C0001.png\"}}'
    )

    from labeling_ui.lib.storage import build_hierarchy

    tree = build_hierarchy(root)
    assert tree[0]["figure_rel_path"].endswith("0001_F02__src.png")
    assert tree[0]["panels"][0]["panel_rel_path"].endswith("0001_F02-P01_chart.png")
    assert tree[0]["panels"][0]["components"][0]["component_type"] == "plot_frame"


def test_build_hierarchy_includes_orphan_figure_entries(tmp_path):
    root = tmp_path / "dataset"
    (root / "figures" / "0001").mkdir(parents=True)
    (root / "panels" / "0001").mkdir(parents=True)
    (root / "links").mkdir(parents=True)

    # 한글 주석: 정상 figure 1개는 파일/인덱스에 모두 존재하게 구성한다.
    (root / "figures" / "0001" / "0001_F01__src.png").write_bytes(b"x")
    (root / "panels" / "0001" / "0001_F01-P01_chart.png").write_bytes(b"x")

    # 한글 주석: orphan 대상 panel은 파일은 있지만 figure_index에는 없는 figure_id를 참조한다.
    (root / "panels" / "0001" / "0001_F99-P01_chart.png").write_bytes(b"x")

    (root / "links" / "component_index.json").write_text("{}", encoding="utf-8")
    (root / "links" / "figure_index.json").write_text(
        __import__("json").dumps(
            {
                "0001_F01": {
                    "figure_id": "0001_F01",
                    "paper_id": "0001",
                    "figure_path": "figures/0001/0001_F01__src.png",
                    "figure_class": None,
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "links" / "panel_index.json").write_text(
        __import__("json").dumps(
            {
                "panels": [
                    {
                        "panel_id": "0001_F01-P01_chart",
                        "paper_id": "0001",
                        "figure_id": "0001_F01",
                        "panel_path": "panels/0001/0001_F01-P01_chart.png",
                    },
                    {
                        "panel_id": "0001_F99-P01_chart",
                        "paper_id": "0001",
                        "figure_id": "0001_F99",
                        "panel_path": "panels/0001/0001_F99-P01_chart.png",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    from labeling_ui.lib.storage import build_hierarchy

    tree = build_hierarchy(root)
    orphan = next(
        item for item in tree if item.get("is_orphan_figure") is True
    )
    assert orphan["paper_id"] == "0001"
    assert orphan["figure_id"] == "0001_F99"
    assert orphan["panels"][0]["panel_rel_path"].endswith(
        "0001_F99-P01_chart.png"
    )
