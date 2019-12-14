#!/usr/local/bin/env python

import os
import shutil
import logging

from .fluorify import Fluorify
from docopt import docopt

logger = logging.getLogger(__name__)

# =============================================================================================
# COMMAND-LINE INTERFACE
# =============================================================================================

usage = """
FLUORIFY
Usage:
  Fluorify [--output_folder=STRING] [--mol_name=STRING] [--ligand_name=STRING] [--complex_name=STRING] [--solvent_name=STRING]
            [--yaml_path=STRING] [--c_atom_list=STRING] [--h_atom_list=STRING] [--num_frames=INT] [--net_charge=INT]
            [--gaff_ver=INT] [--equi=INT] [--num_fep=INT] [--auto_select=STRING] [--charge_only=BOOL] [--vdw_only=BOOL]
            [--num_gpu=INT] [--exclude_dualtopo=BOOL] [--optimize] [--job_type=STRING]...
"""
