# Runs the example program on the image: the runner, the installed program, its manifest, and a refused input.

from oeqa.core.decorator.depends import OETestDepends
from oeqa.runtime.case import OERuntimeTestCase
from oeqa.runtime.decorator.package import OEHasPackage


class HandwiredTest(OERuntimeTestCase):
    PROGRAM = '/usr/share/handwired/heater.hwc'

    @OETestDepends(['ssh.SSHTest.test_ssh'])
    @OEHasPackage(['handwired-run'])
    @OEHasPackage(['handwired-heater'])
    def test_program_runs(self):
        status, output = self.target.run('handwired-run --info %s' % self.PROGRAM)
        self.assertEqual(status, 0, msg=output)
        self.assertIn('heater', output)
        self.assertIn('3 input cells, 2 output cells', output)
        # The same outputs handwired-rs computes on the host for these records.
        status, output = self.target.run("printf '200 21 0\\n215 20 1\\n' | handwired-run %s" % self.PROGRAM)
        self.assertEqual(status, 0, msg=output)
        self.assertEqual(output.strip().splitlines(), ['10 0', '20 0'])

    @OETestDepends(['handwired.HandwiredTest.test_program_runs'])
    def test_manifest_matches(self):
        status, output = self.target.run('cd /usr/share/handwired && sha256sum -c MANIFEST')
        self.assertEqual(status, 0, msg=output)

    @OETestDepends(['handwired.HandwiredTest.test_program_runs'])
    def test_out_of_range_input_is_refused(self):
        status, output = self.target.run("echo '9999 20 0' | handwired-run %s" % self.PROGRAM)
        self.assertNotEqual(status, 0, msg=output)
        self.assertIn('outside the range', output)
