import pytest
from labeling_ui.lib.models import PanelLabel, ComponentLabel


def test_panel_label_required_fields():
    with pytest.raises(Exception):
        PanelLabel()


def test_component_label_required_fields():
    with pytest.raises(Exception):
        ComponentLabel()
