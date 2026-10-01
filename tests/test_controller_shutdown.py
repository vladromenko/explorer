"""The ROS process must stop hardware before closing transport, even on errors."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

class ShutdownTests(unittest.TestCase):
    def run_main(self, exception=None, write_error=False):
        calls=[];driver=SimpleNamespace(send=Mock(side_effect=lambda packet:calls.append('ESTOP')),
            serial=SimpleNamespace(close=lambda:calls.append('close')),destroy_node=lambda:calls.append('destroy'))
        if write_error:driver.send.side_effect=OSError('port lost')
        ros=SimpleNamespace(init=Mock(),spin=Mock(side_effect=exception),try_shutdown=lambda:calls.append('shutdown'))
        signals=SimpleNamespace(SIGTERM=15,SIGINT=2,signal=Mock())
        ns=dict(rclpy=ros,signal=signals,SignalHandlerOptions=SimpleNamespace(NO=0),
            ControllerDriver=lambda:driver,encode=lambda kind:kind,Kind=SimpleNamespace(ESTOP=7),
            serial=SimpleNamespace(SerialException=OSError))
        source=Path(__file__).parents[1]/'src/controller_driver.py'
        tree=ast.parse(source.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        exec(compile(ast.Module(body=[main],type_ignores=[]),str(source),'exec'),ns)
        return ns['main'],calls,signals,ros
    def test_signal_interrupt_leaves_context_valid_until_direct_stop(self):
        main,calls,signals,ros=self.run_main(KeyboardInterrupt())
        main();self.assertEqual(calls,['ESTOP','close','destroy','shutdown'])
        self.assertEqual(ros.init.call_args.kwargs['signal_handler_options'],0)
        self.assertEqual(signals.signal.call_count,2)
    def test_real_runtime_failure_is_not_hidden(self):
        main,calls,_,_=self.run_main(RuntimeError('bad sample'))
        with self.assertRaises(RuntimeError):main()
        self.assertEqual(calls,['ESTOP','close','destroy','shutdown'])
    def test_lost_port_still_closes_context(self):
        main,calls,_,_=self.run_main(KeyboardInterrupt(),True);main()
        self.assertEqual(calls,['close','destroy','shutdown'])
