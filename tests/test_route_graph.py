import pytest
from route_graph import shortest_path


def test_closed_road_reroutes_and_epoch_is_bound():
    graph = {"map_epoch":"m1", "version":2, "nodes":{"a":{},"b":{},"c":{}},
             "edges":[{"from":"a","to":"b","cost":1.,"closed":True},
                      {"from":"a","to":"c","cost":2.}, {"from":"c","to":"b","cost":3.}]}
    route = shortest_path(graph, "a", "b", "m1")
    assert route["places"] == ["a","c","b"]
    assert route["cost"] == 5.
    with pytest.raises(ValueError, match="another map"):
        shortest_path(graph, "a", "b", "m2")
    graph["edges"][1]["closed"] = True
    with pytest.raises(ValueError, match="No open"):
        shortest_path(graph, "a", "b", "m1")
