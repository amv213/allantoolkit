"""
  PHASE.DAT test for allantoolkit (https://github.com/aewallin/allantools)
  Stable32 was used to calculate the deviations we compare against.

  PHASE.DAT comes with Stable32 (version 1.53 was used in this case)

"""

import os
import pytest
import allantoolkit.allantools as allan
import allantoolkit.testutils as testutils

data_file = '../assets/phasedat/PHASE.DAT'
tolerance = 1e-4
verbose = 1


class TestPhaseDat:
    def test_phasedat_adev(self):
        self.generic_test( result= 'phase_dat_adev.txt' , fct= allan.adev, verbose=True )
        
    def test_phasedat_oadev(self):
        self.generic_test( result='phase_dat_oadev.txt' , fct= allan.oadev )
        
    def test_phasedat_mdev(self):
        self.generic_test( result='phase_dat_mdev.txt' , fct= allan.mdev )
        
    def test_phasedat_tdev(self):
        self.generic_test( result='phase_dat_tdev.txt' , fct= allan.tdev )
        
    def test_phasedat_hdev(self):
        self.generic_test( result='phase_dat_hdev.txt' , fct= allan.hdev )
        
    def test_phasedat_ohdev(self):
        self.generic_test( result='phase_dat_ohdev.txt' , fct= allan.ohdev )
        
    def test_phasedat_totdev(self):
        self.generic_test( result='phase_dat_totdev.txt' , fct= allan.totdev )
        
    @pytest.mark.slow
    def test_phasedat_htotdev(self):
        # bias-correction is not implemented, so compare against Stable32-run without bias correction
        self.generic_test( result='phase_dat_htotdev_octave_nobias.txt' , fct= allan.htotdev )
        
    @pytest.mark.slow
    def test_phasedat_mtotdev(self):
        self.generic_test( result='phase_dat_mtotdev_octave.txt' , fct= allan.mtotdev )
        
    @pytest.mark.slow
    def test_phasedat_ttotdev(self):
        self.generic_test( result='phase_dat_ttotdev_octave.txt' , fct= allan.ttotdev )

    def test_phasedat_theo1_alpha0(self):
        self.generic_test( result='phase_dat_theo1_alpha0_decade.txt' , fct= allan.theo1 )
        
    def test_phasedat_mtie(self):
        self.generic_test( result='phase_dat_mtie.txt' , fct= allan.mtie )
        
    def test_phasedat_tierms(self):
        self.generic_test( result='phase_dat_tierms.txt' , fct= allan.tierms )
    
    def generic_test(self, datafile = data_file, result="", fct=None, verbose=False):
        testutils.change_to_test_dir()
        testutils.test_row_by_row( fct, datafile, 1.0, result , verbose=verbose, tolerance=tolerance)