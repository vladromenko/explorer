import threading
from simulator import learning_status as status


def test_slow_archive_count_is_nonblocking_and_single_flight(tmp_path,monkeypatch):
    release=threading.Event();entered=threading.Event();calls=[]
    def slow(*args):
        calls.append(args);entered.set();assert release.wait(2)
        return dict(sock_basket=dict(attempts=33,successful=7))
    monkeypatch.setattr(status,'demonstration_counts',slow)
    try:
        first=status.archive_counts(tmp_path,'fixture')
        assert entered.wait(1)
        assert first['at'] is None and first['pending']
        for _ in range(5):assert status.archive_counts(tmp_path,'fixture')['pending']
        assert len(calls)==1
    finally:release.set()
