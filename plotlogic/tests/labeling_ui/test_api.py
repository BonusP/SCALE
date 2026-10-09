from labeling_ui.app import list_figures, get_hierarchy


def test_list_figures_returns_list():
    res = list_figures()
    assert isinstance(res, list)


def test_get_hierarchy_returns_list():
    res = get_hierarchy()
    assert isinstance(res, list)
