import json
from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'png')


def test_apply_non_chart_suffix_renames_updates_dataset_and_reports(tmp_path: Path) -> None:
    from labeling_ui.lib.non_chart_suffix_migration import apply_non_chart_suffix_renames

    dataset_root = tmp_path / 'data' / 'datasets' / 'phase1_samples'
    docs_root = tmp_path / 'docs' / 'reports'

    _touch(dataset_root / 'papers' / '0001.pdf')
    _touch(dataset_root / 'figures' / '0001' / '0001_F01__src.png')
    _touch(dataset_root / 'figures' / '0001' / '0001_F02__src.png')
    _touch(dataset_root / 'panels' / '0001' / '0001_F01-P01_chart.png')
    _touch(dataset_root / 'panels' / '0001' / '0001_F02-P01_chart.png')
    _touch(dataset_root / 'components' / '0001' / 'crops' / '0001_F01-P01_chart-C01_plot_frame.png')
    _touch(dataset_root / 'components' / '0001' / 'crops' / '0001_F02-P01_chart-C01_plot_frame.png')
    _touch(dataset_root / 'components' / '0001' / 'overlays' / '0001_F01-P01_chart__overlay.png')
    _touch(dataset_root / 'components' / '0001' / 'overlays' / '0001_F02-P01_chart__overlay.png')

    _write_json(
        dataset_root / 'labels' / 'panels' / '0001.json',
        {
            'panels': [
                {
                    'panel_id': '0001_F01-P01_chart',
                    'figure_id': '0001_F01',
                    'panel_bbox_xyxy': [0, 0, 100, 100],
                    'panel_path': 'panels/0001/0001_F01-P01_chart.png',
                    'panel_type': 'chart',
                },
                {
                    'panel_id': '0001_F02-P01_chart',
                    'figure_id': '0001_F02',
                    'panel_bbox_xyxy': [0, 0, 120, 120],
                    'panel_path': 'panels/0001/0001_F02-P01_chart.png',
                    'panel_type': 'chart',
                },
            ]
        },
    )
    _write_json(
        dataset_root / 'labels' / 'components' / '0001.json',
        {
            'components': [
                {
                    'component_id': '0001_F01-P01_chart-C01',
                    'panel_id': '0001_F01-P01_chart',
                    'component_type': 'plot_frame',
                    'component_class': 'chart',
                    'bbox_xyxy': [0, 0, 100, 100],
                    'coord_space': 'panel_px',
                    'figure_class': 'CHART',
                    'panel_path': 'panels/0001/0001_F01-P01_chart.png',
                    'crop_path': 'components/0001/0001_F01-P01_chart-C01_chart.png',
                    'overlay_path': 'components/0001/overlays/0001_F01-P01_chart__overlay.png',
                    'tags': {'auto_generated': True},
                },
                {
                    'component_id': '0001_F02-P01_chart-C01',
                    'panel_id': '0001_F02-P01_chart',
                    'component_type': 'plot_frame',
                    'component_class': 'chart',
                    'bbox_xyxy': [0, 0, 120, 120],
                    'coord_space': 'panel_px',
                    'figure_class': 'CHART',
                    'panel_path': 'panels/0001/0001_F02-P01_chart.png',
                    'crop_path': 'components/0001/0001_F02-P01_chart-C01_chart.png',
                    'overlay_path': 'components/0001/overlays/0001_F02-P01_chart__overlay.png',
                    'tags': {'auto_generated': True},
                },
            ]
        },
    )
    _write_json(
        dataset_root / 'links' / 'gt_plots.json',
        {
            'schema_version': 1,
            'panels': [
                {
                    'panel_id': '0001_F01-P01_chart',
                    'panel_path': 'panels/0001/0001_F01-P01_chart.png',
                    'panel_kind': 'chart',
                    'review_status': 'reject',
                    'gt_bbox_xyxy': None,
                    'annotator': 'A',
                    'note': 'non-chart schematic',
                },
                {
                    'panel_id': '0001_F02-P01_chart',
                    'panel_path': 'panels/0001/0001_F02-P01_chart.png',
                    'panel_kind': 'chart',
                    'review_status': 'reject',
                    'gt_bbox_xyxy': None,
                    'annotator': 'A',
                    'note': 'Suffix 오류, non-chart image',
                },
            ],
        },
    )
    _write_json(
        dataset_root / 'links' / 'gt_axes.json',
        {
            'schema_version': 1,
            'panels': [
                {
                    'panel_id': '0001_F01-P01_chart',
                    'panel_path': 'panels/0001/0001_F01-P01_chart.png',
                    'panel_kind': 'chart',
                    'review_status': 'pending',
                },
                {
                    'panel_id': '0001_F02-P01_chart',
                    'panel_path': 'panels/0001/0001_F02-P01_chart.png',
                    'panel_kind': 'chart',
                    'review_status': 'pending',
                },
            ],
        },
    )
    _write_json(
        docs_root / 'phase2_localize_report.json',
        {
            'panels': [
                {
                    'panel_id': '0001_F01-P01_chart',
                    'panel_path': 'panels/0001/0001_F01-P01_chart.png',
                    'panel_kind': 'chart',
                    'decision': 'REJECT',
                    'evidence': {'predicted_plot_frame_bbox_xyxy': [0, 0, 10, 10]},
                },
                {
                    'panel_id': '0001_F02-P01_chart',
                    'panel_path': 'panels/0001/0001_F02-P01_chart.png',
                    'panel_kind': 'chart',
                    'decision': 'REJECT',
                    'evidence': {'predicted_plot_frame_bbox_xyxy': [0, 0, 10, 10]},
                },
            ]
        },
    )
    _write_json(
        docs_root / 'phase2_gt_axes_seed_v1.json',
        {
            'panels': [
                {
                    'panel_id': '0001_F01-P01_chart',
                    'panel_path': 'panels/0001/0001_F01-P01_chart.png',
                    'panel_kind': 'chart',
                },
                {
                    'panel_id': '0001_F02-P01_chart',
                    'panel_path': 'panels/0001/0001_F02-P01_chart.png',
                    'panel_kind': 'chart',
                },
            ]
        },
    )

    apply_non_chart_suffix_renames(
        dataset_root=dataset_root,
        localize_report_path=docs_root / 'phase2_localize_report.json',
        gt_plots_path=dataset_root / 'links' / 'gt_plots.json',
        gt_axes_path=dataset_root / 'links' / 'gt_axes.json',
        gt_axes_seed_json_path=docs_root / 'phase2_gt_axes_seed_v1.json',
        rename_map={
            '0001_F01-P01_chart': 'schematic',
            '0001_F02-P01_chart': 'image',
        },
    )

    assert not (dataset_root / 'panels' / '0001' / '0001_F01-P01_chart.png').exists()
    assert (dataset_root / 'panels' / '0001' / '0001_F01-P01_schematic.png').exists()
    assert not (dataset_root / 'panels' / '0001' / '0001_F02-P01_chart.png').exists()
    assert (dataset_root / 'panels' / '0001' / '0001_F02-P01_image.png').exists()

    panel_labels = json.loads((dataset_root / 'labels' / 'panels' / '0001.json').read_text(encoding='utf-8'))
    assert [row['panel_id'] for row in panel_labels['panels']] == ['0001_F01-P01_schematic', '0001_F02-P01_image']
    assert [row['panel_type'] for row in panel_labels['panels']] == ['schematic', 'image']

    component_labels = json.loads((dataset_root / 'labels' / 'components' / '0001.json').read_text(encoding='utf-8'))
    by_component = {row['component_id']: row for row in component_labels['components']}
    assert '0001_F01-P01_schematic-C01' in by_component
    assert by_component['0001_F01-P01_schematic-C01']['component_type'] == 'sub_image'
    assert by_component['0001_F01-P01_schematic-C01']['component_class'] == 'schematic'
    assert by_component['0001_F01-P01_schematic-C01']['figure_class'] == 'IMAGE'
    assert by_component['0001_F02-P01_image-C01']['component_type'] == 'image'
    assert by_component['0001_F02-P01_image-C01']['component_class'] == 'image'
    assert by_component['0001_F02-P01_image-C01']['figure_class'] == 'IMAGE'

    gt_plots = json.loads((dataset_root / 'links' / 'gt_plots.json').read_text(encoding='utf-8'))
    assert [row['panel_id'] for row in gt_plots['panels']] == ['0001_F01-P01_schematic', '0001_F02-P01_image']
    assert [row['panel_kind'] for row in gt_plots['panels']] == ['schematic', 'image']

    gt_axes = json.loads((dataset_root / 'links' / 'gt_axes.json').read_text(encoding='utf-8'))
    assert gt_axes['panels'] == []

    localize = json.loads((docs_root / 'phase2_localize_report.json').read_text(encoding='utf-8'))
    assert [row['panel_id'] for row in localize['panels']] == ['0001_F01-P01_schematic', '0001_F02-P01_image']
    assert [row['panel_kind'] for row in localize['panels']] == ['schematic', 'image']

    seed = json.loads((docs_root / 'phase2_gt_axes_seed_v1.json').read_text(encoding='utf-8'))
    assert seed['panels'] == []

    panel_index = json.loads((dataset_root / 'links' / 'panel_index.json').read_text(encoding='utf-8'))
    assert [row['panel_id'] for row in panel_index['panels']] == ['0001_F01-P01_schematic', '0001_F02-P01_image']
    assert [row['candidate_type'] for row in panel_index['panels']] == ['schematic', 'image']

    component_index = json.loads((dataset_root / 'links' / 'component_index.json').read_text(encoding='utf-8'))
    assert sorted(component_index.keys()) == ['0001_F01-P01_schematic-C01', '0001_F02-P01_image-C01']
