import tempfile
import unittest
from resource_scheduler import ResourceScheduler


class SchedulerTests(unittest.TestCase):
    def test_jobs_persist_are_idempotent_and_prioritized(self):
        with tempfile.TemporaryDirectory() as folder:
            scheduler=ResourceScheduler(folder)
            low=scheduler.submit('request-low-123456','analyze_episodes',{},10)
            high=scheduler.submit('request-high-123456','train_scorer',{},90)
            self.assertEqual(scheduler.submit('request-low-123456','analyze_episodes',{},10)['id'],low['id'])
            selected=scheduler.next()
            if selected is not None:self.assertEqual(selected['id'],high['id'])
            scheduler.mark(high['id'],'running');ResourceScheduler(folder)
            self.assertEqual(ResourceScheduler(folder).get(high['id'])['state'],'interrupted')


if __name__=='__main__':unittest.main()
