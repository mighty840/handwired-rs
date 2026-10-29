# Runs the example program on the image: the runner, the installed program, its manifest, refused inputs, its
# signature, and a version log.

from oeqa.core.decorator.depends import OETestDepends
from oeqa.runtime.case import OERuntimeTestCase
from oeqa.runtime.decorator.package import OEHasPackage


class HandwiredTest(OERuntimeTestCase):
    PROGRAM = '/usr/share/handwired/heater.hwc'
    KEY = '/etc/handwired/trusted/example.pub'

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

    @OETestDepends(['handwired.HandwiredTest.test_program_runs'])
    def test_off_grid_input_is_refused(self):
        # Temperatures are whole tenths: half a tenth is between the steps the function declared.
        status, output = self.target.run("echo '200.5 21 0' | handwired-run %s" % self.PROGRAM)
        self.assertNotEqual(status, 0, msg=output)
        self.assertIn('not a multiple of 1', output)

    @OETestDepends(['handwired.HandwiredTest.test_program_runs'])
    @OEHasPackage(['handwired-sign'])
    def test_signature_verifies(self):
        status, output = self.target.run('handwired-sign verify --key %s %s' % (self.KEY, self.PROGRAM))
        self.assertEqual(status, 0, msg=output)
        self.assertIn('slot heater, sequence 1, signed by', output)

    @OETestDepends(['handwired.HandwiredTest.test_signature_verifies'])
    def test_changed_program_is_refused(self):
        # One byte more, as a changed copy would be: the checksum could be redone, the signature can't.
        status, output = self.target.run(
            'cp %s /tmp/changed.hwc && printf x >> /tmp/changed.hwc && '
            'handwired-sign verify --key %s /tmp/changed.hwc %s.sig' % (self.PROGRAM, self.KEY, self.PROGRAM))
        self.assertNotEqual(status, 0, msg=output)
        self.assertIn("doesn't match", output)

    @OETestDepends(['handwired.HandwiredTest.test_signature_verifies'])
    def test_version_log(self):
        log = '/tmp/handwired.log'
        admit = 'handwired-sign admit --log %s --key %s --slot heater %%s' % (log, self.KEY)
        status, output = self.target.run('rm -f %s && handwired-sign log new %s' % (log, log))
        self.assertEqual(status, 0, msg=output)
        status, output = self.target.run(admit % self.PROGRAM)
        self.assertEqual(status, 0, msg=output)
        self.assertIn('installed in slot heater, sequence 1', output)
        # A changed copy, its signature beside it, is refused and logged.
        status, output = self.target.run(
            'cp %s /tmp/changed.hwc && cp %s.sig /tmp/changed.hwc.sig && printf x >> /tmp/changed.hwc && '
            % (self.PROGRAM, self.PROGRAM) + admit % '/tmp/changed.hwc')
        self.assertNotEqual(status, 0, msg=output)
        self.assertIn("doesn't match", output)
        status, output = self.target.run('handwired-sign log show %s' % log)
        self.assertEqual(status, 0, msg=output)
        lines = output.strip().splitlines()
        self.assertTrue(lines[0].endswith(': installed'), msg=output)
        self.assertTrue(lines[1].endswith("refused: the signature doesn't match"), msg=output)
        self.assertIn('2 entries, head ', lines[2])
