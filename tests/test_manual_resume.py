import time
from unittest.mock import Mock
import pytest
from manual_resume import resume_confirmed


def test_waits_for_real_core_confirmation():
    emit = Mock(side_effect=[{"id":"clear"}, {"id":"manual"}])
    read = Mock(side_effect=[{"at":time.time(),"mode":"MANUAL","stop_latched":True},
        {"at":time.time(),"mode":"MANUAL","stop_latched":False,"last_request":{"id":"manual"}}])
    result = resume_confirmed(emit, read, clock=Mock(side_effect=[0, .1, .2]), wait=Mock())
    assert result["core_confirmed"] is True
    assert read.call_count == 2


def test_no_ack_times_out_and_reasserts_stop():
    emit = Mock(return_value={"id":"manual"})
    with pytest.raises(ValueError, match="не подтвердил"):
        resume_confirmed(emit, lambda:{"at":time.time(),"mode":"MANUAL","stop_latched":False},
            clock=Mock(side_effect=[0, .1, 2.]), wait=Mock())
    assert emit.call_args.args == ("stop",)
