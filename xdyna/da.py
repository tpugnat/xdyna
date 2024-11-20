from math import floor
from pathlib import Path
import warnings
import json
import datetime
import time
import tempfile
import sys

from scipy import interpolate, integrate
# from scipy.special import lambertw as W
from scipy.optimize import curve_fit
# from scipy.constants import c as clight
import numpy as np
from numpy.random import default_rng
import pandas as pd

import xobjects as xo
import xtrack as xt
import xpart as xp

# from .postprocess_tools import *
from .postprocess_tools import trapz, simpson, alter_simpson, compute_da_1D, compute_da_2D, compute_da_4D, fit_DA
from .postprocess_tools import _da_raw, _da_smoothing, select_model
from xaux import ProtectFile
from .da_meta import _DAMetaData, _db_access_wait_time, _db_max_lock_time
from .geometry import _bleed, distance_to_polygon_2D
from typing import Union


#_db_access_wait_time = 60 # 0.02 # slow down a lot the waiting time for the access to the AFS database

# TODO: Function to generate DA object from meta file
#       def generate_from_file(filename):
#           DAMetaData(name = child, path= parent)    check that file ends in .meta.json etc




class DA:
    # The real coordinates have to be generated on the machine that will track them (ideally at least).
    # the absolute truth is the sigma; if this translates to slightly different coordinates
    # then that means those machines also have a slightly different closed orbit etc,
    # so the normalised coordinates are still more correct
#     _surv_cols=[
#                 'x_norm_in',  'px_norm_in',  'y_norm_in',  'py_norm_in',  'zeta_in',  'delta_in',
#                 'x_in',       'px_in',   'y_in',   'py_in',
#                 'ang_xy', 'r_xy',
#                 'ang_xpx', 'ang_ypy', 'r_xpxypy',
#                 'xn_out', 'pxn_out', 'yn_out', 'pyn_out', 'zeta_out', 'delta_out',
#                 'nturns', 'paired_to', 'submitted'
#              ]

    def __init__(self, name, *, path=Path.cwd(), use_files: bool=False, read_only: bool=False, **kwargs):
        # Initialise metadata
        self._meta = _DAMetaData(name=name, path=path, use_files=use_files, read_only=read_only)
        if self.meta._new and not read_only:
            self.meta._store_properties = False
            self.meta.nseeds       = kwargs.pop('nseeds',       _DAMetaData._defaults['nseeds'])
            if 'max_turns' not in kwargs.keys():
                raise TypeError("DA.__init__() missing 1 required positional argument: 'max_turns'")
            self.meta.max_turns    = kwargs.pop('max_turns')
            self.meta.min_turns    = kwargs.pop('min_turns',    _DAMetaData._defaults['min_turns'])
            self.meta.energy       = kwargs.pop('energy',       _DAMetaData._defaults['energy'])
            self.meta.db_extension = kwargs.pop('db_extension', _DAMetaData._defaults['db_extension'])
            nemitt_x               = kwargs.pop('nemitt_x',     None)
            nemitt_y               = kwargs.pop('nemitt_y',     None)
            normalised_emittance   = kwargs.pop('normalised_emittance', None)
            if normalised_emittance is not None:
                if nemitt_x is not None or nemitt_y is not None:
                    raise ValueError("Use either normalised_emittance, or nemitt_x and nemitt_y.")
                self.update_emittance(normalised_emittance, update_surv=False)
            elif nemitt_x is not None or nemitt_y is not None:
                if nemitt_x is not None and nemitt_y is not None:
                    self.update_emittance([nemitt_x, nemitt_y], update_surv=False)
                else:
                    raise ValueError("Need both nemitt_x and nemitt_y.")
            self.meta._store()

        # Initialise DA data
        self.memory_threshold = kwargs.pop('memory_threshold', 1e9)
        self._surv = None
        self._da = None
        self._border = None
        self._da_evol = None
        self._da_model = None
        self._active_job = -1
        self._active_job_log = {}
        self._line = None
        self.read_surv()
        self.read_da()
        self.read_border()
        self.read_da_evol()

        # Ignore leftover arguments
        if kwargs != {}:
            print(f"Ignoring unused arguments {list(kwargs.keys())}!")



    # =================================================================
    # ======================= Class attributes ========================
    # =================================================================

    @property
    def survival_data(self):
        if self._surv is None:
            self.read_surv()
        if self._surv is None:  # also nothing on file
            return None

        if self.da_type == 'radial':
            if self.da_dimension == 2:
                view_cols = ['ang_xy', 'r_xy']
            elif self.da_dimension == 3:
                view_cols = ['ang_xy', 'r_xy', 'delta_in']
            elif self.da_dimension == 4:
                view_cols = ['ang_xy', 'ang_xpx', 'ang_ypy', 'r_xpxypy']
            elif self.da_dimension == 5:
                view_cols = ['ang_xy', 'ang_xpx', 'ang_ypy', 'r_xpxypy', 'delta_in']
            elif self.da_dimension == 6:
                view_cols = ['ang_xy', 'ang_xpx', 'ang_ypy', 'r_xpxypy', 'zeta_in', 'delta_in']
        elif self.da_type == 'grid':
            if self.da_dimension == 2:
                view_cols = ['x_norm_in', 'y_norm_in']
            elif self.da_dimension == 3:
                view_cols = ['x_norm_in', 'y_norm_in', 'delta_in']
            elif self.da_dimension == 4:
                view_cols = ['x_norm_in', 'px_norm_in', 'y_norm_in', 'py_norm_in']
            elif self.da_dimension == 5:
                view_cols = ['x_norm_in', 'px_norm_in', 'y_norm_in', 'py_norm_in', 'delta_in']
            elif self.da_dimension == 6:
                view_cols = ['x_norm_in', 'px_norm_in', 'y_norm_in', 'py_norm_in', 'zeta_in', 'delta_in']
        else:
            view_cols = ['x_norm_in', 'px_norm_in', 'y_norm_in', 'py_norm_in', 'zeta_in', 'delta_in']

        if self.meta.nseeds > 0:
            view_cols += ['seed']

        if self.meta.pairs_shift == 0:
            view_cols += ['nturns', 'state']
            df = self._surv[view_cols]
        else:
            orig = self._surv['paired_to'] == self._surv.index
            df = self._surv.loc[orig,view_cols]
            # Change to np.array to ignore index
            df['nturns1'] = np.array(self._surv.loc[orig,'nturns'])
            df['nturns2'] = np.array(self._surv.loc[~orig,'nturns'])
            df['state1'] = np.array(self._surv.loc[orig,'state'])
            df['state2'] = np.array(self._surv.loc[~orig,'state'])
            df['nturns'] = df.loc[:,['nturns1','nturns2']].min(axis=1)
            df['state']  = df.loc[:,['state1','state2']].min(axis=1)

        return df.rename(columns = {
                    'x_norm_in':'x', 'px_norm_in':'px', 'y_norm_in':'y', 'py_norm_in':'py', 'delta_in':'delta', \
                    'ang_xy':'angle', 'ang_xpx':'angle_x', 'ang_ypy':'angle_y', 'r_xy':'amplitude', 'r_xpxypy': 'amplitude' \
                }, inplace=False)


    def to_pandas(self, full: bool=False) -> pd.DataFrame:
        if full:
            return self._surv
        return self.survival_data

    @property
    def meta(self):
        return self._meta

    @property
    def da_type(self):
        return self.meta.da_type

    @property
    def da_border(self):
        return self._lower_border

    @property
    def da_dimension(self):
        return self.meta._da_dim

    @property
    def max_turns(self):
        return self.meta.max_turns

    @max_turns.setter
    def max_turns(self, max_turns: int):
        if max_turns <= self.meta.max_turns:
            print("Warning: new value for max_turns smaller than or equal to existing max_turns! Nothing done.")
            return
        # Need to flag particles that survived to previously defined max_turns as to be resubmitted
        # TODO: need to test
        self.reset_survival_output(mask_by='nturns', mask_func=(lambda x: x == self.meta.max_turns))

    @property
    def nemitt_x(self):
        return self.meta.nemitt_x

    @property
    def nemitt_y(self):
        return self.meta.nemitt_y

    @property
    def normalised_emittance(self):
        if self.nemitt_x is None or self.nemitt_y is None:
            return None
        else:
            return [self.nemitt_x, self.nemitt_y]

    # Not allowed on parallel process
    def update_emittance(self, emit, update_surv: bool=True):
        oldemittance = self.normalised_emittance
        if hasattr(emit, '__iter__'):
            if isinstance(emit, str):
                raise ValueError(f"The emittance has to be a number!")
            elif len(emit) == 2:
                self.meta.nemitt_x = emit[0]
                self.meta.nemitt_y = emit[1]
            elif len(emit) == 1:
                self.meta.nemitt_x = emit[0]
                self.meta.nemitt_y = emit[0]
            else:
                raise ValueError(f"The emittance must have one or two values (for nemitt_x and nemitt_y)!")
        else:
            self.meta.nemitt_x = emit
            self.meta.nemitt_y = emit
        # Recalculate initial conditions if set
        if update_surv and self._surv is not None and oldemittance is not None and self.normalised_emittance != oldemittance:
            print("Updating emittance")
            corr_x = np.sqrt( oldemittance[0] / self.nemitt_x )
            corr_y = np.sqrt( oldemittance[1] / self.nemitt_y )
            if self.surv_exists():
                with ProtectFile(self.meta.surv_file, 'r+b', wait=_db_access_wait_time) as pf:
                    self.read_surv(pf)
                    self._surv['x_norm_in']  *= corr_x
                    self._surv['px_norm_in'] *= corr_x
                    self._surv['y_norm_in']  *= corr_y
                    self._surv['py_norm_in'] *= corr_y
                    self._surv['r_xy'] = np.sqrt( self._surv['x_norm_in']**2 + self._surv['y_norm_in']**2 )
                    self._surv['r_xpxypy'] = np.sqrt( self._surv['x_norm_in']**2 + self._surv['px_norm_in']**2 \
                                                    + self._surv['y_norm_in']**2 + self._surv['py_norm_in']**2 )
                    self.write_surv(pf)
            elif self._surv is not None:
                self._surv['x_norm_in']  *= corr_x
                self._surv['px_norm_in'] *= corr_x
                self._surv['y_norm_in']  *= corr_y
                self._surv['py_norm_in'] *= corr_y
                self._surv['r_xy'] = np.sqrt( self._surv['x_norm_in']**2 + self._surv['y_norm_in']**2 )
                self._surv['r_xpxypy'] = np.sqrt( self._surv['x_norm_in']**2 + self._surv['px_norm_in']**2 \
                                                + self._surv['y_norm_in']**2 + self._surv['py_norm_in']**2 )

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, line):
        if line is None:
            if self.meta.line_file == -1:
                self.meta.line_file = None  # The line is no longer manually added
            self._line = None
            self.meta.energy = None
        else:
            self.meta.line_file = -1  # Mark line as manually added
            self._line = line
            if self.meta.nseeds == 0:
                # No seeds, so self.line is just the line
                if not isinstance(line, xt.Line):
                    self._line = xt.Line.from_dict(line)
                self.meta.energy = self.line.particle_ref.p0c[0]
            else:
                # Seeds, so line is a dict of lines
                energy = []
                for seed, this_line in line.items():
                    if not isinstance(this_line, xt.Line):
                        self._line[seed] = xt.Line.from_dict(this_line)
                    energy = [ *energy, self._line[seed].particle_ref.p0c[0] ]
                if len(np.unique([ round(e, 4) for e in energy])) != 1:
                    raise ValueError(f"The lines for the different seeds have different energies: {energy}!")
                self.meta.energy = energy[0]

    @property
    def madx_file(self):
        return self.meta.madx_file

    @madx_file.setter
    def madx_file(self, file):
        self.meta.madx_file = file

    @property
    def line_file(self):
        if self.meta.line_file == -1:
            return 'line manually added'
        else:
            return self.meta.line_file

    @line_file.setter
    def line_file(self, file):
        self.meta.line_file = file

    def load_line_from_file(self, file=None):
        if file is not None and self.line_file is not None and self.line_file != -1 and file != self.line_file:
            raise ValueError("Trying to load different line than the one specified in the metafile! " + \
                             "This is not allowed. If you want to remove the line from the metafile, " + \
                             "please do DA.line_file = None.")
        if self.line_file == -1:
            # Remove existing line that was added manually
            self.line = None
        if file is None:
            if self.line_file is None:
                print("Warning: no file given and no file specified in the metafile! No line loaded.")
                return
            else:
                file = self.line_file
        file = Path(file)
        if not file.exists():
            raise ValueError(f"Line file {file.as_posix()} not found!")

        with ProtectFile(file, 'r', wait=_db_access_wait_time, max_lock_time=_db_max_lock_time) as pf:
            line = json.load(pf)

        # Fix string keys in JSON: seeds are int
        if self.meta.nseeds > 0:
            if len(line.keys()) > self.meta.nseeds:
                raise ValueError("Line file not compatible with seeds! Expected a dict of lines with seeds as keys.")
            line = {int(seed): xt.Line.from_dict(l) for seed, l in line.items()}
        else:
            line = xt.Line.from_dict(line)
        self._line = line
        self.line_file = file



    # =================================================================
    # ================ Generation of intial conditions ================
    # =================================================================

    def _prepare_generation(self, normalised_emittance=None, nseeds=None, pairs_shift: float=0, pairs_shift_var=None):
        # Does survival already exist?
        if self.survival_data is not None:
            print("Warning: Initial conditions already exist! No generation done.")
            return

        # Avoid writing to the meta at every property
        self.meta._store_properties = False

        # If non-default values are specified, copy them to the metadata
        # In the grid generation further below, only the metadat values
        # should be used (and not the function ones)!
        if normalised_emittance is not None:
            self.update_emittance(normalised_emittance, update_surv=False)
        if self.normalised_emittance is None:
            raise ValueError("No emittance defined! Do this first before generating initial conditions")
        if nseeds is not None:
            self.meta.nseeds = nseeds
        if pairs_shift != 0:
            self.meta.pairs_shift = pairs_shift
            if pairs_shift_var is None:
                raise ValueError("Need to set coordinate for the shift between pairs with pairs_shift_var!")
            else:
                self.meta.pairs_shift_var = pairs_shift_var
        elif pairs_shift_var is not None:
            raise ValueError("Need to set magnitude of shift between pairs with pairs_shift!")

    def _create_pairs(self):
        # Create pairs if requested
        if self.meta.pairs_shift != 0:
            coord = self.meta.pairs_shift_var
            # Rename the coordinate correctly
            if coord in ['x', 'y', 'px', 'py']:
                coord += '_norm_in'
                recalc = 'rang'
            elif coord in ['zeta', 'delta']:
                coord += '_in'
            elif coord in ['angle', 'r']:
                coord = coord[:3] + '_xy'
                recalc = 'xy'
            else:
                raise ValueError(f"Value '{self.meta.pairs_shift_var}' not allowed for pairs_shift_var!")
            # The paired particles will be individual rows, with a pointer
            # to the index value of the particle they originate from.
            self._surv['paired_to'] = self._surv.index
            df = self._surv.copy()
            # Add the shift, and recalculate x and y if needed.
            df[coord] += self.meta.pairs_shift
            if recalc == 'xy':
                df['x_norm_in'] = df['r_xy']*np.cos(df['ang_xy']*np.pi/180)
                df['y_norm_in'] = df['r_xy']*np.sin(df['ang_xy']*np.pi/180)
                # Skip recalculation of r and ang, to not screw up analysis of DA later
            # Join existing and new dataframe
            df.set_index(df.index + len(df.index), inplace=True)
            self._surv = pd.concat([self._surv, df])


    def set_coordinates(self, *,
                        x: Union[float,list,None]=None, px: Union[float,list,None]=None, 
                        y: Union[float,list,None]=None, py: Union[float,list,None]=None, 
                        zeta: Union[float,list,None]=None, delta: Union[float,list,None]=None,
                        normalised_emittance: Union[float,list,None]=None, nseeds: Union[int,None]=None, 
                        pairs_shift: float=0, pairs_shift_var: Union[str,None]=None):
        """Let user provide initial coordinates for each plane. Those data are saved in _surv.

        Args:
            x (float | list | None, optional):                    Horizontal normalised position in [sigma]. Defaults to None.
            px (float | list | None, optional):                   Horizontal normalised momentum. Defaults to None.
            y (float | list | None, optional):                    Vertical normalised position in [sigma]. Defaults to None.
            py (float | list | None, optional):                   Vertical normalised momentum. Defaults to None.
            zeta (float | list | None, optional):                 Longitudinal phase. Defaults to None.
            delta (float | list | None, optional):                Longitudinal dispersion in energy. Defaults to None.
            normalised_emittance (float | list | None, optional): Normalised emittances. Defaults to None.
            nseeds (int | None, optional):                        Total number of seed to simulate. Defaults to None.
            pairs_shift (float, optional):                        Shift step length for pair particles. Ignored if equal to 0. Defaults to 0.
            pairs_shift_var (str | None, optional):               Direction in which pair particles are shifted. Defaults to None.
        """

        self._prepare_generation(normalised_emittance, nseeds, pairs_shift, pairs_shift_var)

        user_provided_coords = [i for i in [x,px, y, py, zeta, delta] if i is not None]

        # check that all provided lists have the same length
        assert len({len(i) for i in user_provided_coords}) == 1, 'Mismatch in length of provided lists'

        # replace all unused variables with zero
        x, px, y, py, zeta, delta = [ i if i is not None else 0 for i in [x,px, y, py, zeta, delta]]

        # Make all combinations
        if self.meta.nseeds > 0:
            seeds = np.arange(1, self.meta.nseeds+1)
            x, px, y, py, zeta, delta, seeds = np.array(np.meshgrid(x, px, y, py, zeta, delta, seeds)).reshape(7,-1)

        # Make dataframe
        self._surv = pd.DataFrame()
        if self.meta.nseeds > 0:
            self._surv['seed'] = seeds.astype(int)
        self._surv['nturns'] = -1
        self._surv['x_norm_in'] = x
        self._surv['y_norm_in'] = y
        self._surv['px_norm_in'] = px
        self._surv['py_norm_in'] = py
        self._surv['zeta_in'] = zeta
        self._surv['delta_in'] = delta
        self._surv['x_out'] = -1
        self._surv['y_out'] = -1
        self._surv['px_out'] = -1
        self._surv['py_out'] = -1
        self._surv['zeta_out'] = -1
        self._surv['delta_out'] = -1
        self._surv['s_out'] = -1
        self._surv['state'] = 1
        self._surv['submitted'] = False
        self._surv['finished'] = False
        self._create_pairs()
        self.write_surv()
        self.meta.da_type = 'free'
        self.meta.da_dim = len(user_provided_coords)
        # self.meta.r_max = np.max(np.sqrt(x**2 + y**2))
        self.meta.npart = len(self._surv.index)
        self.meta._store()



    # Not allowed on parallel process
    def generate_initial_grid(self, *, x_min: float, x_max: float, x_step: Union[float,None]=None, x_num: Union[int,None]=None,
                                y_min: float, y_max: float, y_step: Union[float,None]=None, y_num: Union[int,None]=None,
                                px_norm: Union[list,float]=0, py_norm: Union[list,float]=0, zeta: Union[list,float]=0, 
                                delta: Union[list,float]=0.00027, normalised_emittance: Union[list,float,None]=None, nseeds: Union[int,None]=None, 
                                pairs_shift: float=0, pairs_shift_var: Union[str,None]=None):
        """Generate the initial conditions in a 2D X-Y grid.

        Args:
            x_min (float):                                        Min range in the x-plan in [sigma].
            x_max (float):                                        Max range in the x-plan in [sigma].
            y_min (float):                                        Min range in the y-plan in [sigma].
            y_max (float):                                        Max range in the y-plan in [sigma].
            x_step (float | None, optional):                      Amplitude step size in the x-plan in [sigma]. Defaults to None.
            x_num (int | None, optional):                         Number of step in the x-plan in amplitude. Defaults to None.
            y_step (float | None, optional):                      Amplitude step size in the y-plan in [sigma]. Defaults to None.
            y_num (int | None, optional):                         Number of step in the y-plan in amplitude. Defaults to None.
            px_norm (list | float, optional):                     Horizontal normalised momentum. Defaults to 0.
            py_norm (list | float, optional):                     Vertical normalised momentum. Defaults to 0.
            zeta (list | float, optional):                        Longitudinal phase. Defaults to 0.
            delta (list | float, optional):                       Longitudinal momentum. Defaults to 0.00027.
            normalised_emittance (list | float | None, optional): Normalised emittances. Defaults to None.
            nseeds (int | None, optional):                        Total number of seed to simulate. Defaults to None.
            pairs_shift (float,  optional):                       Shift step length for pair particles. Ignored if equal to 0. Defaults to 0.
            pairs_shift_var (str | None, optional):               Direction in which pair particles are shifted. Defaults to None.
        """

        self._prepare_generation(normalised_emittance, nseeds, pairs_shift, pairs_shift_var)

        # Make the grid in xy
        def check_options(coord_min, coord_max, coord_step, coord_num, plane):
            if coord_step is None and coord_num is None:
                raise ValueError(f"Specify at least '{plane}_step' or '{plane}_num'.")
            elif coord_step is not None and coord_num is not None:
                raise ValueError(f"Use only one of '{plane}_step' and '{plane}_num', not both.")
            elif coord_step is not None:
                coord_num = floor( (coord_max-coord_min) / coord_step ) + 1
                coord_max = coord_min + (coord_num-1) * coord_step
            return coord_min, coord_max, coord_num

        x_min, x_max, x_num = check_options(x_min, x_max, x_step, x_num, 'x')
        y_min, y_max, y_num = check_options(y_min, y_max, y_step, y_num, 'y')

        x_space = np.linspace(x_min, x_max, x_num)
        y_space = np.linspace(y_min, y_max, y_num)

        # Make all combinations
        if self.meta.nseeds > 0:
            seeds = np.arange(1, self.meta.nseeds+1)
            x, y, seeds = np.array(np.meshgrid(x_space, y_space, seeds)).reshape(3,-1)
        else:
            x, y = np.array(np.meshgrid(x_space, y_space)).reshape(2,-1)

        # Make dataframe
        self._surv = pd.DataFrame()
        if self.meta.nseeds > 0:
            self._surv['seed'] = seeds.astype(int)
        self._surv['ang_xy'] = np.arctan2(y,x)*180/np.pi
        self._surv['r_xy'] = np.sqrt(x**2 + y**2)
        self._surv['nturns'] = -1
        self._surv['x_norm_in'] = x
        self._surv['y_norm_in'] = y
        self._surv['px_norm_in'] = px_norm
        self._surv['py_norm_in'] = py_norm
        self._surv['zeta_in'] = zeta
        self._surv['delta_in'] = delta
        self._surv['x_out'] = -1
        self._surv['y_out'] = -1
        self._surv['px_out'] = -1
        self._surv['py_out'] = -1
        self._surv['zeta_out'] = -1
        self._surv['delta_out'] = -1
        self._surv['s_out'] = -1
        self._surv['state'] = 1
        self._surv['submitted'] = False
        self._surv['finished'] = False
        self._create_pairs()
        self.write_surv()
        self.meta.da_type = 'grid'
        self.meta.da_dim = 2
        self.meta.r_max = np.max(np.sqrt(x**2 + y**2))
        self.meta.ang_min = np.min(np.arctan2(y,x)*180/np.pi)
        self.meta.ang_max = np.max(np.arctan2(y,x)*180/np.pi)
        self.meta.npart = len(self._surv.index)
        self.meta._store()



    # Not allowed on parallel process
    def generate_initial_radial(self, *, angles: int, r_min: float, r_max: float, r_step: Union[float,None]=None, r_num: Union[int,None]=None, 
                                ang_min: float=0, ang_max: float=90, px_norm: Union[list,float]=0, py_norm: Union[list,float]=0,
                                zeta: Union[list,float]=0, delta: Union[list,float]=0.00027, 
                                normalised_emittance: Union[list,float,None]=None, nseeds: Union[int,None]=None, 
                                pairs_shift: float=0, pairs_shift_var: Union[str,None]=None, open_border: bool=True):
        """Generate the initial conditions in a 2D polar grid.

        Args:
            angles (int):                                         Number of angles per seed.
            r_min (float):                                        Min range of amplitude in [sigma].
            r_max (float):                                        Max range of amplitude in [sigma]
            r_step (float | None, optional):                      Amplitude step size in [sigma]. Defaults to None.
            r_num (int | None, optional):                         Number of step in amplitude. Defaults to None.
            ang_min (float, optional):                            Lower range of the angular distribution in [deg]. Defaults to 0.
            ang_max (float, optional):                            Upper range of the angular distribution in [deg]. Defaults to 90.
            px_norm (list | float, optional):                     Horizontal normalised momentum. Defaults to 0.
            py_norm (list | float, optional):                     Vertical normalised momentum. Defaults to 0.
            zeta (list | float, optional):                        Longitudinal phase. Defaults to 0.
            delta (list | float, optional):                       Longitudinal momentum. Defaults to 0.00027.
            normalised_emittance (list | float | None, optional): Normalised emittances. Defaults to None.
            nseeds (int | None, optional):                        Total number of seed to simulate. Defaults to None.
            pairs_shift (float,  optional):                       Shift step length for pair particles. Ignored if equal to 0. Defaults to 0.
            pairs_shift_var (str | None, optional):               Direction in which pair particles are shifted. Defaults to None.
            open_border (bool, optional):                         If True, the 1st and last angles will be `ang_min+ang_step` and `ang_max-ang_step` respectivel. Defaults to True.
        """

        self._prepare_generation(normalised_emittance, nseeds, pairs_shift, pairs_shift_var)

        # Make the grid in r
        if r_step is None and r_num is None:
            raise ValueError("Specify at least 'r_step' or 'r_num'.")
        elif r_step is not None and r_num is not None:
            raise ValueError("Use only one of 'r_step' and 'r_num', not both.")
        elif r_step is not None:
            r_num = floor( (r_max-r_min) / r_step ) + 1
            r_max = r_min + (r_num-1) * r_step
        r = np.linspace(r_min, r_max, r_num )
        # Make the grid in angles
        ang_step = (ang_max-ang_min) / (angles+1)
        ang = np.linspace(ang_min+ang_step*open_border, ang_max-ang_step*open_border, angles )
        # Make all combinations
        if self.meta.nseeds > 0:
            seeds = np.arange(1,self.meta.nseeds+1)
            ang, seeds, r = np.array(np.meshgrid(ang, seeds, r)).reshape(3,-1)
        else:
            r, ang = np.array(np.meshgrid(r, ang)).reshape(2,-1)
        # Get the normalised coordinates
        x = r*np.cos(ang*np.pi/180)
        y = r*np.sin(ang*np.pi/180)
        # Make dataframe
        self._surv = pd.DataFrame()
        if self.meta.nseeds > 0:
            self._surv['seed'] = seeds.astype(int)
        self._surv['ang_xy'] = ang
        self._surv['r_xy'] = r
        self._surv['nturns'] = -1
        self._surv['x_norm_in'] = x
        self._surv['y_norm_in'] = y
        self._surv['px_norm_in'] = px_norm
        self._surv['py_norm_in'] = py_norm
        self._surv['zeta_in'] = zeta
        self._surv['delta_in'] = delta
        self._surv['x_out'] = -1
        self._surv['y_out'] = -1
        self._surv['px_out'] = -1
        self._surv['py_out'] = -1
        self._surv['zeta_out'] = -1
        self._surv['delta_out'] = -1
        self._surv['s_out'] = -1
        self._surv['state'] = 1
        self._surv['submitted'] = False
        self._surv['finished'] = False
        self._create_pairs()
        self.write_surv()
        self.meta.da_type = 'radial'
        self.meta.da_dim = 2
        self.meta.r_max = r_max
        self.meta.ang_min = ang_min
        self.meta.ang_max = ang_max
        self.meta.npart = len(self._surv.index)
        self.meta._store()



    # Not allowed on parallel process
    def generate_random_initial(self, *, num_part: int=1000, r_max: float=25, px_norm: Union[list,float]=0, py_norm: Union[list,float]=0, 
                                zeta: Union[list,float]=0, delta: Union[list,float]=0.00027, ang_min: float=0, ang_max: float=90, 
                                normalised_emittance: Union[list,float, None]=None, nseeds: Union[int,None]=None, 
                                pairs_shift: float=0, pairs_shift_var: Union[str,None]=None):
        """Generate the initial conditions in a 2D random grid.

        Args:
            num_part (int, optional):                             Number of particle per seed. Defaults to 1000.
            r_max (float, optional):                              Max range of amplitude. Defaults to 25.
            px_norm (list | float, optional):                     Horizontal normalised momentum. Defaults to 0.
            py_norm (list | float, optional):                     Vertical normalised momentum. Defaults to 0.
            zeta (list | float, optional):                        Longitudinal phase. Defaults to 0.
            delta (list | float, optional):                       Longitudinal momentum. Defaults to 0.00027.
            ang_min (float, optional):                            Lower range of the angular distribution in [deg]. Defaults to 0.
            ang_max (float, optional):                            Upper range of the angular distribution in [deg]. Defaults to 90.
            normalised_emittance (list | float | None, optional): Normalised emittances. Defaults to None.
            nseeds (int | None, optional):                        Total number of seed to simulate. Defaults to None.
            pairs_shift (float,  optional):                       Shift step length for pair particles. Ignored if equal to 0. Defaults to 0.
            pairs_shift_var (str | None, optional):               Direction in which pair particles are shifted. Defaults to None.
        """

        self._prepare_generation(normalised_emittance, nseeds, pairs_shift, pairs_shift_var)

        # Make the data
        rng = default_rng()
#         ang_min = 0 if ang_min is None else ang_min
#         ang_max = 90 if ang_max is None else ang_max
        if self.meta.nseeds > 0:
            r = rng.uniform(low=0, high=r_max**2, size=num_part*self.meta.nseeds)
            th = rng.uniform(low=ang_min, high=ang_max*np.pi/180, size=num_part*self.meta.nseeds)
            seeds = np.repeat(np.arange(1,self.meta.nseeds+1), num_part)
        else:
            r = rng.uniform(low=0, high=r_max**2, size=num_part)
            th = rng.uniform(low=ang_min, high=ang_max*np.pi/180, size=num_part)
        r = np.sqrt(r)
        x = r*np.cos(th)
        y = r*np.sin(th)

        # Make dataframe
        self._surv = pd.DataFrame()
        if self.meta.nseeds > 0:
            self._surv['seed'] = seeds.astype(int)
        self._surv['x_norm_in'] = x
        self._surv['y_norm_in'] = y
        self._surv['nturns'] = -1
        self._surv['px_norm_in'] = px_norm
        self._surv['py_norm_in'] = py_norm
        self._surv['zeta_in'] = zeta
        self._surv['delta_in'] = delta
        self._surv['x_out'] = -1
        self._surv['y_out'] = -1
        self._surv['px_out'] = -1
        self._surv['py_out'] = -1
        self._surv['zeta_out'] = -1
        self._surv['delta_out'] = -1
        self._surv['s_out'] = -1
        self._surv['state'] = 1
        self._surv['submitted'] = False
        self._surv['finished'] = False
        self._create_pairs()
        self.write_surv()
        self.meta.da_type = 'monte_carlo'
        self.meta.da_dim = 2
        self.meta.r_max = r_max
        self.meta.ang_min = ang_min
        self.meta.ang_max = ang_max
        self.meta.npart = len(self._surv.index)
        self.meta._store()



    # Not allowed on parallel process
    def add_random_initial(self, *, num_part: int=5000, min_turns=None):
        from .ml import MLBorder

        # TODO: make compatible with seeds and with pairs
        if self.meta.nseeds > 0 or self.meta.pairs_shift != 0:
            raise NotImplementedError

        # TODO: erase already calculated DA values

        # Set minimum turns
        if min_turns is None:
            if self.meta.min_turns is None:
                self.meta.min_turns = 20
        else:
            if self.meta.min_turns is not None and self.meta.min_turns != min_turns:
                # TODO: This should only be checked if we already re-sampled.
                # There is no harm by changing min_turns after the initial run
                print(f"Warning: 'min_turns' can be set only once (and is already set to {self.meta.min_turns}). "
                      + "Ignored the new value.")

        # Get existing results
        if self.survival_data is None:
            raise ValueError("No survival data found!")
        data = self.survival_data
        # TODO: check that all px_norm etc are the same
        px_norm = np.unique(self._surv['px_norm_in'])[0]
        py_norm = np.unique(self._surv['py_norm_in'])[0]
        zeta = np.unique(self._surv['zeta_in'])[0]
        delta = np.unique(self._surv['delta_in'])[0]

        # Get minimum and maximum borders
        prev = time.process_time()
        print(f"Getting minimum border (at Nmin={self.meta.min_turns} turns)... ", end='')
        ML_min = MLBorder(data.x, data.y, data.nturns, at_turn=self.meta.min_turns, memory_threshold=self.memory_threshold)
        ML_min.fit(50)
        ML_min.evaluate(0.5)
        min_border_x, min_border_y = ML_min.border
        print(f"done (in {round(time.process_time()-prev,2)} seconds).")

        prev = time.process_time()
        print(f"Getting maximum border (at Nmax={self.meta.max_turns} turns)... ", end='')
        ML_max = MLBorder(data.x, data.y, data.nturns, at_turn=self.meta.max_turns, memory_threshold=self.memory_threshold)
        ML_max.fit(50)
        ML_max.evaluate(0.5)
        max_border_x, max_border_y = ML_max.border
        print(f"done (in {round(time.process_time()-prev,2)} seconds).")

        # TODO: catch when border tells us that not enough samples are available and larger r or larger Nmin is needed

        print("Generating samples... ", end='')
        # First we create a 'pool' of samples that contains 100 times as many particles as wished.
        # This is to get a reliable distribution to select from later.
        # We make this pool by sampling a uniform ring, from the inner circle of the max_border (with bleed)
        # until the outer circle of the min_border (with bleed). Then we veto particles if they're not
        # within the borders (with bleed)
        rng = default_rng()
        _, r_max, bleed_min_val = _bleed(min_border_x, min_border_y, margin=0.1)
        r_min, _, bleed_max_val = _bleed(max_border_x, max_border_y, margin=0.1)

        r_min -= bleed_max_val
        r_max += bleed_min_val
        r_max = min(r_max, self.meta.r_max)

        n_pool = num_part * 100
        x_pool = np.array([])
        y_pool = np.array([])
        d_pool = np.array([])

        # Go in chunks of maximum ram during calculation of distances
        n_samples = int(self.memory_threshold / 25 / max(len(min_border_x), len(max_border_x)))

        while n_pool > 0:
            n_samples = min(n_samples, int(2.5*n_pool))
            r = rng.uniform(low=r_min**2, high=r_max**2, size=n_samples)
            r = np.sqrt(r)
            th = rng.uniform(low=0, high=2*np.pi, size=n_samples)
            x = r*np.cos(th)
            y = r*np.sin(th)
            distance_to_min = distance_to_polygon_2D(x, y, min_border_x, min_border_y)
            distance_to_max = distance_to_polygon_2D(x, y, max_border_x, max_border_y)

            # We only sample points in between min_border and max_border, but also allow
            # for a bit of 'bleed' over the border.
            between   = (distance_to_min <= 0) & (distance_to_max > 0)
            bleed_min = (distance_to_min > 0) & (distance_to_min < bleed_min_val)
            bleed_max = (distance_to_max <= 0) & (distance_to_max > -bleed_max_val)

            x_clean = x[bleed_max | bleed_min | between]
            y_clean = y[bleed_max | bleed_min | between]

            # Rescale the distances to adapt to the probabilities later:
            distance = distance_to_max.copy()
            # Those that bleeded over the maximum border are rescaled between 0 (on the border) and -1/2 (maximum bleed)
            distance[bleed_max] /= bleed_max_val*2
            # Those in between are rescaled to the distance between the two borders: 0 (on the max border), 1 (on the min border)
            distance[between] /= distance_to_max[between] - distance_to_min[between]
            # Those that bleeded over the minimum border are rescaled between 1 (on the border) and 3/2 (maximum bleed)
            distance[bleed_min] = distance_to_min[bleed_min]/bleed_min_val/2 + 1

            d_clean = distance[bleed_max | bleed_min | between]

            n_pool -= len(x_clean)
            if n_pool < 0:
                x_pool = np.concatenate([x_pool, x_clean[:n_pool]])
                y_pool = np.concatenate([y_pool, y_clean[:n_pool]])
                d_pool = np.concatenate([d_pool, d_clean[:n_pool]])
            else:
                x_pool = np.concatenate([x_pool, x_clean])
                y_pool = np.concatenate([y_pool, y_clean])
                d_pool = np.concatenate([d_pool, d_clean])

        # The pool is generated, now we sample!
        # As a probability distribution, we use a gaussian scaled by a square root, shifted 1/2 to the left
        prob = np.sqrt(d_pool+0.5)*np.exp(-(d_pool+0.5)**2)
        prob /= prob.sum()
        ids = rng.choice(np.arange(len(x_pool)), size=num_part, replace=False, p=prob, shuffle=False)
        x_new = x_pool[ids]
        y_new = y_pool[ids]
        print(f"done (in {round(time.process_time()-prev,2)} seconds).")

        # Make new dataframe
        df_new = pd.DataFrame()
#         if self.meta.nseeds > 0:
#             self._surv['seed'] = seeds.astype(int)
        df_new['x_norm_in'] = x_new
        df_new['y_norm_in'] = y_new
        df_new['nturns'] = -1
        df_new['px_norm_in'] = px_norm
        df_new['py_norm_in'] = py_norm
        df_new['zeta_in'] = zeta
        df_new['delta_in'] = delta
        df_new['x_out'] = -1
        df_new['y_out'] = -1
        df_new['px_out'] = -1
        df_new['py_out'] = -1
        df_new['zeta_out'] = -1
        df_new['delta_out'] = -1
        df_new['s_out'] = -1
        df_new['state'] = 1
        df_new['submitted'] = False
        df_new['finished'] = False
#         self._create_pairs()   # Needs to only act on non-finished particles ...
        self._surv = pd.concat([
                            self._surv,
                            df_new
                    ], ignore_index=True)
        self.write_surv()



    # =================================================================
    # =================== create line from MAD-X ======================
    # =================================================================

    # TODO: can we get particle mass from mask??? Particle type??
    # Allowed on parallel process
    def build_line_from_madx(self, sequence:str, *, file=None, apertures: bool=False, errors: bool=True, \
                             mass: float=xp.PROTON_MASS_EV, store_line: bool=True, seeds:Union[list,None]=None, 
                             other_madx_flag:dict={}, run_all_seeds: bool=False, raise_madx_critical_warning: bool=True):
        """_summary_

        Args:
            sequence (str):                               Sequence to load from MAD-X.
            file (str | Path | None, optional):           Madx mask file. Defaults to None.
            apertures (bool, optional):                   If true, load the aperture from madx. Defaults to False.
            errors (bool, optional):                      If true, load the magnet errors from madx. Defaults to True.
            store_line (bool, optional):                  If true, save line in a file. Defaults to True.
            seeds (list | None, optional):                In case of multiseed simulation, specify the seeds to runs here. Defaults to None.
            other_madx_flag (dict, optional):             Specify flags other than `%SEEDRAN` that need to be replaced within the mask. Defaults to {}.
            run_all_seeds (bool, optional):               In case of multiseed simulation, set to True if you want to run all seeds at once. Defaults to False.
            raise_madx_critical_warning (bool, optional): If True, raise the critical warning from MAD-X. Defaults to True.
        """
        # seeds=None  ->  find seed automatically (on parallel process)
        from cpymad.madx import Madx
        if file is None: 
            if self.madx_file is None:
                raise ValueError("No MAD-X file specified in arguments nor in metadata. Cannot build line.")
        else:
            if self.madx_file is None:
                self.madx_file = file
            elif self.madx_file != file:
                self.madx_file = file
                print("Warning: MAD-X file specified in arguments does not match with metadata. " + \
                      "Used the former and updated the metadata.")

        if store_line and self.line_file is None:
            self.line_file = self.madx_file.parent / (self.madx_file.stem + '.line.json')

        # Find which seeds to run, and initialise line file if it is to be stored
        if self.meta.nseeds == 0:
            seeds = [ None ]
        else:
            self._line = {}
            if run_all_seeds:
                if seeds is not None:
                    raise ValueError("Cannot have run_all_seeds=True and seeds!=None. Choose one.")
                seeds = range(1, self.meta.nseeds+1)
                print("Calculating the line from MAD-X for all available seeds. Make sure this is not ran on " + \
                      "a parallel process, as results might be unpredictable and probably wrong.")
                if store_line:
                    if self.line_file.exists():
                        print(f"Warning: line_file {self.line_file} exists and is being overwritten.")
                    data = {str(seed): 'Running MAD-X' for seed in seeds}
                    with ProtectFile(self.line_file, 'w', wait=_db_access_wait_time) as pf:
                        json.dump(data, pf, cls=xo.JEncoder, indent=True)
            else:
                if seeds is not None:
                    if not hasattr(seeds, '__iter__'):
                        seeds = [ seeds ]
                elif not store_line:
                    raise ValueError("Cannot determine seed automatically (run_all_seeds=False and seed=None) " + \
                                     "if store_line=False, as the line file is needed to find seeds for which " + \
                                     "the MAD-X calculation isn't yet performed.")
                if store_line:
                    # Hack to make it thread-safe
                    created = False
                    if not self.line_file.exists():
                        try:
                            if seeds is None:
                                seeds = [ 1 ]
                            data = {str(seed): 'Running MAD-X' for seed in seeds}
                            with ProtectFile(self.line_file, 'x', wait=_db_access_wait_time) as fid:
                                json.dump(data, fid, cls=xo.JEncoder, indent=True)
                            created = True
                        except FileExistsError:
                            pass
                    if not created:
                        with ProtectFile(self.line_file, 'r+', wait=_db_access_wait_time) as pf:
                            data = json.load(pf)
                            if seeds is None:
                                n_existing = len(data.keys())
                                if n_existing == self.meta.nseeds:
                                    # TODO: implement system to rerun failed MAD-X
#                                     to_redo = False
#                                     for seed, val in data.items():
#                                         if val == 'MAD-X failed':
#                                             seeds = [ int(seed) ]
#                                             to_redo = True
#                                             break
#                                     if not to_redo:
                                    print("File already has a line for every seed. Nothing to be done.")
                                    return
                                else:
                                    seeds = [ n_existing + 1 ]
                            for seed in seeds:
                                if str(seed) in data.keys():
                                    print(f"Warning: line_file {self.line_file} already contained a result for seed {seed}. " + \
                                           "This is being overwritten.")
                                data[str(seed)] = 'Running MAD-X'
                            pf.truncate(0)  # Delete file contents (to avoid appending)
                            pf.seek(0)      # Move file pointer to start of file
                            json.dump(data, pf, cls=xo.JEncoder, indent=True)    
                print(f"Processing seed{'s' if len(seeds) > 1 else ''} {seeds[0] if len(seeds) == 1 else seeds}.")

        # Run MAD-X
        energy = []
        for seed in seeds:
            with tempfile.TemporaryDirectory() as tmpdir:
                if seed is None:
                    madin = self.madx_file
                    madout = self.madx_file.parent / (self.madx_file.stem + '.madx.out')
                else:
                    madin = Path(tmpdir) / self.madx_file.name
                    madout = self.madx_file.parent / (self.madx_file.stem + '.' + str(seed) + '.madx.out')
                    with ProtectFile(self.madx_file, 'r', wait=_db_access_wait_time,
                                     max_lock_time=_db_max_lock_time) as fin:
                        data = fin.read()
                        data = data.replace('%SEEDRAN', str(seed))
                        for kk,vv in other_madx_flag.items():
                            data = data.replace(kk, str(vv))
                        with ProtectFile(madin, 'w', wait=_db_access_wait_time) as fout:
                            fout.write(data)
                with ProtectFile(madout, 'w', wait=_db_access_wait_time) as pf:
                    print(f"Running MAD-X in {tmpdir}")
                    mad = Madx(stdout=pf)
                    mad.chdir(tmpdir)
                    mad.call(madin.as_posix())
                    # TODO: check if MAD-X failed: if so, write 'MAD-X failed' into line_file
                # TODO: check if madx raised a message stating a variable was not defined before being used
                with ProtectFile(madout, 'r', wait=_db_access_wait_time) as pf:
                    lls = pf.readlines()
                    error_madx = ''
                    for l in lls:
                        if "++++++ warning: illegal expression, expression NOT updated for the following :" in l:
                            error_madx += l
                    if error_madx != '':
                        if raise_madx_critical_warning:
                            raise ValueError(f"MAD-X raised the following critical warning:\n{error_madx}")
                        else:
                            if seed is None:
                                print(f"MAD-X raised the following critical warning for {self.madx_file.stem}.madx.out:\n{error_madx}\n")
                            else:
                                print(f"MAD-X raised the following critical warning for {self.madx_file.stem}.{seed}.madx.out:\n{error_madx}\n")
                            
            line = xt.Line.from_madx_sequence(mad.sequence[sequence], apply_madx_errors=errors, \
                                              install_apertures=apertures)
            line.particle_ref = xp.Particles(mass0=mass, gamma0=mad.sequence[sequence].beam.gamma)
            energy = [ *energy, line.particle_ref.p0c[0] ]

            # Save result
            if seed is None:
                self._line = line
                if store_line:
                    with ProtectFile(self.line_file, 'w', wait=_db_access_wait_time) as pf:
                        json.dump(self.line.to_dict(), pf, cls=xo.JEncoder, indent=True)
            else:
                if self._line is None:
                    self._line = {}
                self._line[seed] = line
                if store_line:
                    with ProtectFile(self.line_file, 'r+', wait=_db_access_wait_time) as pf:
                        data = json.load(pf)
                        data[str(seed)] = line.to_dict()
                        pf.truncate(0)  # Delete file contents (to avoid appending)
                        pf.seek(0)      # Move file pointer to start of file
                        json.dump(data, pf, cls=xo.JEncoder, indent=True)

        # Check energy
        if self.meta.energy is not None:
            energy = [ *energy, self.meta.energy ]
        if len(np.unique([ round(e, 4) for e in energy])) != 1:
            raise ValueError(f"The lines for the different seeds have different energies: {energy}!")
        if self.meta.energy is None:
            self.meta.energy = energy[0]



    # =================================================================
    # ==================== Xtrack tracking jobs =======================
    # =================================================================

    # Allowed on parallel process
    def track_job(self, *,  npart: Union[int,None]=None, logging: bool=True, force_single_seed_per_job: Union[bool,None]=None, 
                  with_progress: bool=False, co_search_at: Union[str,None]=None):
        """Track the particles in the lines and automatically manage the seeds.

        Args:
            npart (Union[int,None], optional):                 Max number of particle to track. Defaults to None.
            logging (bool, optional):                          Add logging in the submission file. Defaults to True.
            force_single_seed_per_job (bool | None, optional): If True, force to track only one seed per job. Defaults to None.
            with_progress (bool, optional):                    Show progress while tracking. Defaults to False.
            co_search_at (str | None, optional):               Change the starting point at which the CO is computed, example: `ip7`. Defaults to None.
        """
        if self.line is None:
            if self.meta.line_file is not None and self.meta.line_file != -1 and self.meta.line_file.exists():
                self.load_line_from_file()
            else:
                raise Exception("No line loaded nor found on file!")

        # Sometimes the CO is hard to find. Changing the starting point can help and it does not change anything for the tracking!
        if co_search_at is not None:
            if self.meta.nseeds == 0:
                self.line.twiss_default['co_search_at'] = co_search_at
            else:
                for seed in self.line:
                     self.line[seed].twiss_default['co_search_at'] = co_search_at

        # Create a job: get job ID and start logging
        part_ids, seeds, flag = self._create_job(npart, logging, force_single_seed_per_job)
        if flag != 0:
            return
        job_id = str(self._active_job)

        # Define tracking procedure
#         def track_per_seed(context, tracker, x_norm, y_norm, px_norm, py_norm, zeta, delta, nemitt_x, nemitt_y, nturn):
        def track_per_seed(context, line, x_norm, y_norm, px_norm, py_norm, zeta, delta, nemitt_x, nemitt_y, nturn, with_progress):
            
            # openmp context and radiation do not play nicely together, so temp. switch to single thread context
            if line._context.openmp_enabled:
                line.discard_tracker()
                line.build_tracker(_context=xo.ContextCpu())
            # Create initial particles
            part = line.build_particles(
                                      x_norm=x_norm, y_norm=y_norm, px_norm=px_norm, py_norm=py_norm, zeta=zeta, delta=delta,
                                      nemitt_x=nemitt_x, nemitt_y=nemitt_y
                                     )
            if line._context.openmp_enabled:
                line.build_tracker(_context=context)
#             tracker.track(particles=part, num_turns=nturn)
            line.track(particles=part, num_turns=nturn, with_progress=with_progress)
            context.synchronize()
            return part

        if self.meta.nseeds == 0:
            # Build tracker(s) if not yet done
            if self.line.tracker is None:
                print("Building tracker.")
                self.line.build_tracker()

            # Select initial particles
            context = self.line.tracker._buffer.context
            x_norm  = self._surv.loc[part_ids, 'x_norm_in'].to_numpy()
            y_norm  = self._surv.loc[part_ids, 'y_norm_in'].to_numpy()
            px_norm = self._surv.loc[part_ids, 'px_norm_in'].to_numpy()
            py_norm = self._surv.loc[part_ids, 'py_norm_in'].to_numpy()
            zeta    = self._surv.loc[part_ids, 'zeta_in'].to_numpy()
            delta   = self._surv.loc[part_ids, 'delta_in'].to_numpy()

            self._append_job_log('output', datetime.datetime.now().isoformat() + '  Start tracking job ' + str(job_id) + '.', logging=logging)
#             part=track_per_seed(context,self.line.tracker,
            part=track_per_seed(context,self.line,
                                x_norm, y_norm, px_norm, py_norm, zeta, delta, 
                                self.nemitt_x, self.nemitt_y, self.meta.max_turns, with_progress=with_progress)
            self._append_job_log('output', datetime.datetime.now().isoformat() + '  Done tracking job ' + str(job_id) + '.', logging=logging)

            # Store results
            part_id   = context.nparray_from_context_array(part.particle_id)
            sort      = np.argsort(part_id)
            x_out     = context.nparray_from_context_array(part.x)[sort]
            y_out     = context.nparray_from_context_array(part.y)[sort]
            survturns = context.nparray_from_context_array(part.at_turn)[sort]
            px_out    = context.nparray_from_context_array(part.px)[sort]
            py_out    = context.nparray_from_context_array(part.py)[sort]
            zeta_out  = context.nparray_from_context_array(part.zeta)[sort]
            delta_out = context.nparray_from_context_array(part.delta)[sort]
            s_out     = context.nparray_from_context_array(part.s)[sort]
            state     = context.nparray_from_context_array(part.state)[sort]

            if self.surv_exists():
                with ProtectFile(self.meta.surv_file, 'r+b', wait=_db_access_wait_time) as pf:
                    self.read_surv(pf)
                    self._surv.loc[part_ids, 'finished'] = True
                    self._surv.loc[part_ids, 'x_out'] = x_out
                    self._surv.loc[part_ids, 'y_out'] = y_out
                    self._surv.loc[part_ids, 'nturns'] = survturns.astype(np.int64)
                    self._surv.loc[part_ids, 'px_out'] = px_out
                    self._surv.loc[part_ids, 'py_out'] = py_out
                    self._surv.loc[part_ids, 'delta_out'] = delta_out
                    self._surv.loc[part_ids, 's_out'] = s_out
                    self._surv.loc[part_ids, 'state'] = state
                    self.write_surv(pf)
            else:
                self._surv.loc[part_ids, 'finished'] = True
                self._surv.loc[part_ids, 'x_out'] = x_out
                self._surv.loc[part_ids, 'y_out'] = y_out
                self._surv.loc[part_ids, 'nturns'] = survturns.astype(np.int64)
                self._surv.loc[part_ids, 'px_out'] = px_out
                self._surv.loc[part_ids, 'py_out'] = py_out
                self._surv.loc[part_ids, 'zeta_out'] = zeta_out
                self._surv.loc[part_ids, 'delta_out'] = delta_out
                self._surv.loc[part_ids, 's_out'] = s_out
        else:
            if 0 in self.line.keys():
                # Line file dict is 0-indexed
                seeds = [ seed-1 for seed in seeds ]
            for seed in seeds:
                # Build tracker(s) if not yet done
                if self.line[seed].tracker is None:
                    print(f"Building tracker for seed {seed}.")
                    self.line[seed].build_tracker()

                # Select initial particles
                context = self.line[seed].tracker._buffer.context
                part_ids_seed = part_ids[self._surv.loc[part_ids, 'seed']==seed]
                x_norm  = self._surv.loc[part_ids_seed, 'x_norm_in'].to_numpy()
                y_norm  = self._surv.loc[part_ids_seed, 'y_norm_in'].to_numpy()
                px_norm = self._surv.loc[part_ids_seed, 'px_norm_in'].to_numpy()
                py_norm = self._surv.loc[part_ids_seed, 'py_norm_in'].to_numpy()
                zeta    = self._surv.loc[part_ids_seed, 'zeta_in'].to_numpy()
                delta   = self._surv.loc[part_ids_seed, 'delta_in'].to_numpy()

                self._append_job_log('output', datetime.datetime.now().isoformat() + '  Start tracking job ' + str(job_id) + '.', logging=logging)
                part=track_per_seed(context,self.line[seed],
                                    x_norm, y_norm, px_norm, py_norm, zeta, delta, 
                                    self.nemitt_x, self.nemitt_y, self.meta.max_turns,with_progress=with_progress)

                self._append_job_log('output', datetime.datetime.now().isoformat() + '  Done tracking job ' + str(job_id) + '.', logging=logging)

                # Store results
                part_id   = context.nparray_from_context_array(part.particle_id)
                sort      = np.argsort(part_id)
                x_out     = context.nparray_from_context_array(part.x)[sort]
                y_out     = context.nparray_from_context_array(part.y)[sort]
                survturns = context.nparray_from_context_array(part.at_turn)[sort]
                px_out    = context.nparray_from_context_array(part.px)[sort]
                py_out    = context.nparray_from_context_array(part.py)[sort]
                zeta_out  = context.nparray_from_context_array(part.zeta)[sort]
                delta_out = context.nparray_from_context_array(part.delta)[sort]
                s_out     = context.nparray_from_context_array(part.s)[sort]
                state     = context.nparray_from_context_array(part.state)[sort]

                if self.surv_exists():
                    with ProtectFile(self.meta.surv_file, 'r+b', wait=_db_access_wait_time) as pf:
                        self.read_surv(pf)
                        self._surv.loc[part_ids_seed, 'finished'] = True
                        self._surv.loc[part_ids_seed, 'x_out'] = x_out
                        self._surv.loc[part_ids_seed, 'y_out'] = y_out
                        self._surv.loc[part_ids_seed, 'nturns'] = survturns.astype(np.int64)
                        self._surv.loc[part_ids_seed, 'px_out'] = px_out
                        self._surv.loc[part_ids_seed, 'py_out'] = py_out
                        self._surv.loc[part_ids_seed, 'zeta_out'] = zeta_out
                        self._surv.loc[part_ids_seed, 'delta_out'] = delta_out
                        self._surv.loc[part_ids_seed, 's_out'] = s_out
                        self._surv.loc[part_ids_seed, 'state'] = state
                        self.write_surv(pf)
                else:
                    self._surv.loc[part_ids_seed, 'finished'] = True
                    self._surv.loc[part_ids_seed, 'x_out'] = x_out
                    self._surv.loc[part_ids_seed, 'y_out'] = y_out
                    self._surv.loc[part_ids_seed, 'nturns'] = survturns.astype(np.int64)
                    self._surv.loc[part_ids_seed, 'px_out'] = px_out
                    self._surv.loc[part_ids_seed, 'py_out'] = py_out
                    self._surv.loc[part_ids_seed, 'zeta_out'] = zeta_out
                    self._surv.loc[part_ids_seed, 'delta_out'] = delta_out
                    self._surv.loc[part_ids_seed, 's_out'] = s_out

        self._update_job_log({
            'finished_time': datetime.datetime.now().isoformat(),
            'status': 'Finished'
        }, logging=logging)


    # NOT allowed on parallel process!
    def resubmit_unfinished(self):
        if self.surv_exists():
            with ProtectFile(self.meta.surv_file, 'r+b', wait=_db_access_wait_time) as pf:
                self.read_surv(pf)
                mask = self._surv['finished'] == False
                self._surv.loc[mask, 'submitted'] = False
                self.write_surv(pf)
        else:
            print("No survival file found! Nothing done.")

    # TODO: need to test
    def reset_survival_output(self, mask=None, mask_by=None, mask_func=None):
        if mask is None:
            if mask_by is None:
                if mask_func is None:
                    mask = slice(None,None,None)
                else:
                    raise ValueError("Cannot use 'mask_func' without 'mask_by'!")
            else:
                if mask_func is None:
                    raise ValueError("Need to use 'mask_func' when using 'mask_by'!")
        elif mask_by is not None or mask_func is not None:
            raise ValueError("Have to choose between using 'mask=..', or 'mask_by=..' and 'mask_func=..'!")

        def _reset_surv(mask):
            self._surv.loc[mask, 'x_out'] = -1
            self._surv.loc[mask, 'y_out'] = -1
            self._surv.loc[mask, 'px_out'] = -1
            self._surv.loc[mask, 'py_out'] = -1
            self._surv.loc[mask, 'zeta_out'] = -1
            self._surv.loc[mask, 'delta_out'] = -1
            self._surv.loc[mask, 's_out'] = -1
            self._surv.loc[mask, 'nturns'] = -1
            self._surv.loc[mask, 'state'] = 1
            self._surv.loc[mask, 'submitted'] = False
            self._surv.loc[mask, 'finished'] = False

        if self.surv_exists():
            with ProtectFile(self.meta.surv_file, 'r+b', wait=_db_access_wait_time) as pf:
                self.read_surv(pf)
                if mask is None:
                    mask = mask_func(self._surv[mask_by])
                _reset_surv(mask)
                self.write_surv(pf)
        elif self._surv is not None:
            if mask is None:
                mask = mask_func(self._surv[mask_by])
            _reset_surv(mask)


    # =================================================================
    # ========================= Calculate DA ==========================
    # =================================================================

    @property
    def t_steps(self):
        if self.survival_data is None:
            raise ValueError('Run the simulation before using plot_particles.')
#         data=self.survival_data
#         if self.meta.nseeds == 0:
#             return {0: np.unique(data['nturns'].values)}
#         else:
#             return {s: np.unique(data.loc[data['seed']==s,'nturns'].values) for s in range(1,self.meta.nseeds+1)}
        if self.meta.nseeds == 0:
            return {0: np.unique(self._da['t'].values)}
        else:
            return {s: np.unique(self._da.loc[self._da['seed']==s,'t'].values) for s in range(1,self.meta.nseeds+1)}

    def da(self, t=None, seed=None):
        return self._get_da_prop(t=t, seed=seed, prop='DA lower')

    def da_min(self, t=None, seed=None):
        return self._get_da_prop(t=t, seed=seed, prop='DAmin lower')

    def da_max(self, t=None, seed=None):
        return self._get_da_prop(t=t, seed=seed, prop='DAmax lower')

    def da_lower(self, t=None, seed=None):
        return self._get_da_prop(t=t, seed=seed, prop='DA lower')

    def da_min_lower(self, t=None, seed=None):
        return self._get_da_prop(t=t, seed=seed, prop='DAmin lower')

    def da_max_lower(self, t=None, seed=None):
        return self._get_da_prop(t=t, seed=seed, prop='DAmax lower')

    def da_upper(self, t=None, seed=None):
        return self._get_da_prop(t=t, seed=seed, prop='DA upper')

    def da_min_upper(self, t=None, seed=None):
        return self._get_da_prop(t=t, seed=seed, prop='DAmin upper')

    def da_max_upper(self, t=None, seed=None):
        return self._get_da_prop(t=t, seed=seed, prop='DAmax upper')

    def da_projection(self, t=None):
        if t is None:
            t = self.max_turns
        if self.meta.nseeds == 0:
            lturns = self.t_steps
            closest_t = max(lturns[0][lturns[0] <= t])
            da = self._da[(self._da.t == closest_t)]
            return da['DAmin lower'].values, da['DA lower'].values, da['DAmax lower'].values

        else:
            lturns = self.t_steps
            da = pd.DataFrame([],index=range(1,self.meta.nseeds+1), columns=self._da.columns)
            for ss in range(1,self.meta.nseeds+1):
                closest_t = max(lturns[ss][lturns[ss] <= t])
                da.loc[ss,self._da.columns] = self._da[(self._da.t == closest_t) & (self._da.seed == ss)].values
            return da['DAmin lower'].min(), da['DA lower'].mean(), da['DAmax lower'].max(), \
                   da.loc[:,['seed','DA lower']].rename(columns = {'DA lower':'DA'})

    def border(self, t=None, seed=None):
        ang = self._get_da_prop(t=t, seed=seed, prop='ang lower', data=self._border, \
                              enforce_single=False)
        amp = self._get_da_prop(t=t, seed=seed, prop='amp lower', data=self._border, \
                              enforce_single=False)
        # TODO: 4D, 5D, 6D ...
        return [ang, amp]

    def border_lower(self, t=None, seed=None):
        ang = self._get_da_prop(t=t, seed=seed, prop='ang lower', data=self._border, \
                              enforce_single=False)
        amp = self._get_da_prop(t=t, seed=seed, prop='amp lower', data=self._border, \
                              enforce_single=False)
        # TODO: 4D, 5D, 6D ...
        return [ang, amp]

    def border_upper(self, t=None, seed=None):
        ang = self._get_da_prop(t=t, seed=seed, prop='ang upper', data=self._border, \
                              enforce_single=False)
        amp = self._get_da_prop(t=t, seed=seed, prop='amp upper', data=self._border, \
                              enforce_single=False)
        # TODO: 4D, 5D, 6D ...
        return [ang, amp]

    def _get_da_prop(self, t, seed, prop, data=None, enforce_single=True):
        ss = "" if seed is None else f"and seed {seed}"
        if self.meta.nseeds == 0:
            if seed is not None:
                raise ValueError("Cannot use 'seed' as this DA object has no seeds!")
            seed = 0
        elif seed is None:
            raise ValueError("Need to specify a seed.")

        if t is None:
            t = self.max_turns
        elif t not in self.t_steps[seed]:
            # TODO interpolate times in between
            raise NotImplementedError

        if data is None:
            data = self._da
        result = data.loc[(data['seed']==seed) & (data['t']==t), prop]
#         result = data.loc[('seed'==seed) & ('t'==t), prop] # TODO: There is a bug used like that
        if enforce_single:
            if len(result) > 1:
                raise ValueError(f"Found multiple values for {prop} at time {t}{ss}.")
            return result.values[0]
        else:
            return result.values


    # Not allowed on parallel process
#     def get_lower_da(self,at_turn=None,seed=None):
#         '''Return the DA lower estimation at a specific turn in the form of a DataFrame with the following columns:
#         ['turn','border','avg','min','max']

#     Parameters
#     ----------
#     at_turn: Turn at which this estimation must be computed (Default=max_turns).
#     seed:    For multiseed simulation, the seed must be specified (Default=None).
# '''

#         if at_turn is None:
#             at_turn=self.max_turns
#         if self._da is None:
#             self.calculate_da(at_turn=at_turn,smoothing=True)
#         if self.meta.nseeds!=0 and seed is None:
#             raise ValueError('Please specify the seed number for multiseeds simulation.')

#         if self.meta.nseeds==0:
#             davsturns=self._lower_davsturns
#         else:
#             davsturns=self._lower_davsturns[seed]
#         return davsturns.loc[davsturns.loc[davsturns.turn<=at_turn,'turn'].astype(float).idxmax(),:]


#     # Not allowed on parallel process
#     def get_upper_da(self,at_turn=None,seed=None):
#         '''Return the DA upper estimation at a turn in the form of a DataFrame with the following columns:
#         ['turn','border','avg','min','max']

#     Parameters
#     ----------
#     at_turn: Turn at which this estimation must be computed (Default=max_turns).
#     seed:    For multiseed simulation, the seed must be specified (Default=None).
# '''

#         if at_turn is None:
#             at_turn=self.max_turns
#         if self._da is None:
#             self.calculate_da(at_turn=at_turn,smoothing=True)
#         if self.meta.nseeds!=0 and seed is None:
#             raise ValueError('Please specify the seed number for multiseeds simulation.')

#         if self.meta.nseeds==0:
#             davsturns=self._upper_davsturns
#         else:
#             davsturns=self._upper_davsturns[seed]
#         return davsturns.loc[davsturns.loc[davsturns.turn<=at_turn,'turn'].astype(float).idxmax(),:]


    def _set_da_and_border(self,seed,at_turn,border_min,border_max,ang_range):
        new_da=pd.DataFrame({'DA lower':    [compute_da_1D(border_min['angle'], border_min['amplitude'], ang_range)],
                             'DAmin lower': [min(border_min['amplitude'])],
                             'DAmax lower': [max(border_min['amplitude'])],
                             'DA upper':    [compute_da_1D(border_max['angle'], border_max['amplitude'], ang_range)],
                             'DAmin upper': [min(border_max['amplitude'])],
                             'DAmax upper': [max(border_max['amplitude'])]
                            })
        self._da=concat(self._da,new_da,seed=seed,t=at_turn)

        self._border=concat(self._border,
                            border_min.rename(columns={'angle':'ang lower', 'amplitude':'amp lower', 'id':'id lower'}, 
                                              inplace=False),
                            seed=seed,t=at_turn)

        self._border=concat(self._border,
                            border_max.rename(columns={'angle':'ang upper', 'amplitude':'amp upper', 'id':'id upper'}, 
                                              inplace=False),
                            seed=seed,t=at_turn)


    # Not allowed on parallel process
    # TODO: Fix description to fit the new data format
    def calculate_da(self, at_turn: Union[int,None]=None, angular_precision: int=10, smoothing: bool=True, list_seed: Union[list,None]=None,
                     interp_order: str='1D', interp_method: str='trapz', force: bool=False, save: bool=True):
        """Compute the DA upper and lower estimation at a specific turn in the form of a pandas table ['turn','border','avg','min','max'] or for multiseeds { seed:['turn','border','avg','min','max'], 'stat':['turn','avg','min','max'] }
        
        Warning:
            The borders might change after using postprocess as it imposes turn-by-turn monoticity.

        Args:
            at_turn (int | None, optional):    Turn at which this estimation must be computed. Defaults to max_turns.
            angular_precision (int, optional): Angular precision in [deg.] for the raw estimation of the borders.  It is better to use high value in order to minimise the risk of catching stability island. Defaults to 10.
            smoothing (bool, optional):        True in order to smooth the borders for the random particle distribution. Defaults to True.
            list_seed (list | None, optional): List of seeds on which the da will be estimated. Defaults to None.
            interp_order (str, optional):      Interpolation order for the DA average: '1D', '2D', '4D'. Defaults to '1D'.
            interp_method (str, optional):     Interpolation method for the DA average: 'trapz', 'simpson', 'alternative_simpson'. Defaults to 'trapz'.
            force (bool, optional):            Flag for reseting the da and border DataFrames. Defaults to False.
            save (bool, optional):             Flag for saving da and border DataFrames in files. Defaults to True.
        """

        if self.survival_data is None:
            raise ValueError('Run the simulation before using plot_particles.')

        if interp_order=='1D':
            compute_da=compute_da_1D
        elif interp_order=='2D':
            compute_da=compute_da_2D
        elif interp_order=='4D':
            compute_da=compute_da_4D
        else:
            raise ValueError("interp_order must be either: '1D', '2D' or '4D'!")
            
        if interp_method=='trapz':
            interp=trapz
        elif interp_method=='simpson':
            interp=simpson
        elif interp_method=='alternative_simpson':
            interp=alter_simpson
        else:
            raise ValueError("interp_method must be either: 'trapz', 'simpson', 'alternative_simpson'!")

        # Initialize input and da array
        if at_turn is None:
            at_turn=self.max_turns
        if at_turn > self.max_turns:
            raise ValueError(f'at_turn cannot be higher than the max number of turns for the simulation, here max_turn={DA.max_turns}')

        if force:
            self._da = None
            self._border = None
        elif self._da is None or self._border is None:
            self.read_da()
            self.read_border()
        if self._da is None:
            self._da=pd.DataFrame(columns=['t','seed', 'DA lower','DAmin lower','DAmax lower',
                                           'DA upper','DAmin upper','DAmax upper'])

            self._border=pd.DataFrame(columns=['t','seed', 'id lower','id upper'])
        if at_turn in np.unique(self._da['t']):
            return

        # Select data per seed if needed
        data=self.survival_data.copy()
        data['id']= data.index
        if self.da_type == 'radial':
            data['round_angle']= data['angle']

        elif self.da_type in ['grid', 'monte_carlo', 'free']:
            data['angle']      = np.arctan2(data['y'],data['x'])*180/np.pi
            data['amplitude']  = np.sqrt(data['x']**2+data['y']**2)
            data['round_angle']= np.floor(data['angle']/angular_precision)*angular_precision
        if self.meta.nseeds==0:
            list_seed=[ 0 ]
            list_data=[ data ]
            list_new_da     = np.array([None])
            list_new_border = np.array([None])
        else:
            if list_seed is None or len(list_seed)==0:
                list_seed=[s for s in range(1,self.meta.nseeds+1)]
            list_data=[ data[data.seed==s].copy() for s in list_seed ]
            list_new_da     = np.array([None]  * self.meta.nseeds)
            list_new_border = np.array([None]  * self.meta.nseeds)
        ang_range=(self.meta.ang_min,self.meta.ang_max)
        mask_da     = [True] * len(self._da)
        mask_border = [True] * len(self._border)

        # Run DA raw border detection
#         sys.stdout.write(f'seed {0:>3d}')
        idx = 0
        for ss,dd in zip(list_seed,list_data):
#             sys.stdout.write(f'\rseed {seed:>3d}')

            # Get DA raw estimation
#             border_min, border_max =_da_raw(dd,at_turn,seed,compute_da, ang_range)
            list_new_da[idx], list_new_border[idx] =_da_raw(dd, at_turn, ss, compute_da, ang_range)

            if self.da_type == 'radial' and list_new_border[idx].isnull().values.any():
                print('Warning: the sample was too small and this estimation will be biased ' \
                      +f'(seed {ss}, at_turn {ss}, nb missing values: {list_new_border[idx].isnull().values.sum()})!')

            mask_da     = mask_da     & ~( ( self._da.seed    ==ss ) & ( self._da.t    ==at_turn ) )
            mask_border = mask_border & ~( ( self._border.seed==ss ) & ( self._border.t==at_turn ) )
            idx += 1

#             # Detect range to look at the DA border
#             losses =dd.nturns<at_turn
#             loss=dd.loc[ losses,:]; min_loss=min(loss['amplitude'])
#             surv=dd.loc[~losses,:]; max_surv=max(surv['amplitude'])
#             min_amplitude = min([min_loss,max_surv])-2
#             max_amplitude = max([min_loss,max_surv])+2

#             # Get a raw DA estimation from losses
#             border_max={'id':[],'angle':[],'amplitude':[]}
#             border_min={'id':[],'angle':[],'amplitude':[]}
#             for ang in np.unique(dd['round_angle']):
#                 # Select angulare slice
#                 section=dd.loc[dd.round_angle==ang,:]

#                 # Identify losses and surviving particles
#                 losses =section.nturns<at_turn
#                 section_loss=section.loc[ losses,:]; section_loss=section_loss.loc[section_loss['amplitude']<=max_amplitude,:]
#                 section_surv=section.loc[~losses,:]; section_surv=section_surv.loc[section_surv['amplitude']>=min_amplitude,:]

#                 # Detect DA boundary
#                 if not section_loss.empty and not section_surv.empty:
#                     min_amplitude_loss=min(section_loss['amplitude'])
#                     border_max['amplitude'].append(min_amplitude_loss)
#                     border_max['angle'].append(section_loss.loc[section_loss['amplitude']==min_amplitude_loss,'angle'].values[0])
#                     border_max['id'].append(section_loss.loc[section_loss['amplitude']==min_amplitude_loss,'id'].values[0])

#                     mask = section_surv['amplitude']<min_amplitude_loss
#                     if any(mask):
#                         max_amplitude_surv=max(section_surv['amplitude'][mask])

#                         border_min['amplitude'].append(max_amplitude_surv)
#                         border_min['angle'].append(section_surv.loc[section_surv['amplitude']==max_amplitude_surv,'angle'].values[0])
#                         border_min['id'].append(section_surv.loc[section_surv['amplitude']==max_amplitude_surv,'id'].values[0])

#                 elif not section_loss.empty:
#                     min_amplitude_loss=min(section_loss['amplitude'])
#                     border_max['amplitude'].append(min_amplitude_loss)
#                     border_max['angle'].append(section_loss.loc[section_loss['amplitude']==min_amplitude_loss,'angle'].values[0])
#                     border_max['id'].append(section_loss.loc[section_loss['amplitude']==min_amplitude_loss,'id'].values[0])

#                 elif not section_surv.empty:
#                     max_amplitude_surv=max(section_surv['amplitude'])

#                     border_min['amplitude'].append(max_amplitude_surv)
#                     border_min['angle'].append(section_surv.loc[section_surv['amplitude']==max_amplitude_surv,'angle'].values[0])
#                     border_min['id'].append(section_surv.loc[section_surv['amplitude']==max_amplitude_surv,'id'].values[0])

#             border_max=pd.DataFrame(border_max)
#             border_min=pd.DataFrame(border_min)

            if self.da_type in ['monte_carlo', 'free']:

                losses =dd.nturns<at_turn
                loss=dd.loc[ losses,:]
                surv=dd.loc[~losses,:]

                # Check if losses lower than upper DA boundary, add those to max DA boundary
                border_max_fit=fit_DA(border_max['angle'], border_max['amplitude'], ang_range)
                loss_in_DA_max = loss.loc[loss['amplitude']< border_max_fit(loss['angle']),:]
                if not loss_in_DA_max.empty:
                    border_max=pd.concat([border_max,loss_in_DA_max])

                # Check if lower DA boundary cross upper DA boundary, remove problematic dot from lower DA boundary
                border_max_fit=fit_DA(border_max['angle'], border_max['amplitude'], ang_range)
                border_min=border_min.loc[border_min['amplitude']<border_max_fit(border_min['angle']),:]

                # Check if losses lower than min DA boundary
                border_min_fit=fit_DA(border_min['angle'], border_min['amplitude'], ang_range)
                pb_border_max=border_max.loc[border_max['amplitude']<=border_min_fit(border_max['angle']),:]
                while not pb_border_max.empty:
                    for idx,ploss in pb_border_max.iterrows():
                        border_min_fit=fit_DA(border_min['angle'], border_min['amplitude'], ang_range)
                        
                        if ploss['amplitude'] <= border_min_fit(ploss['angle']):
                            lower=border_min.loc[border_min['angle']==max(border_min['angle'][border_min['angle']<ploss['angle']]),:]
                            upper=border_min.loc[border_min['angle']==min(border_min['angle'][border_min['angle']>ploss['angle']]),:]
                            lower_amp=lower['amplitude'].tolist()[0] ; lower_ang=lower['angle'].tolist()[0]
                            upper_amp=upper['amplitude'].tolist()[0] ; upper_ang=upper['angle'].tolist()[0]

                            # Remove lower border point too high for the losses
                            if lower_amp < upper_amp:
                                border_min.drop(index=upper.index.values, inplace=True)
                                upper=border_min.loc[border_min['angle']==min(border_min['angle'][border_min['angle']>ploss['angle']]),:]
                                upper_amp=upper['amplitude'].tolist()[0] ; upper_ang=upper['angle'].tolist()[0]

                            else:
                                border_min.drop(index=lower_amp, inplace=True)
                                lower=border_min.loc[border_min['angle']==max(border_min['angle'][border_min['angle']<ploss['angle']]),:]
                                lower_amp=lower['amplitude'].tolist()[0] ; lower_ang=lower['angle'].tolist()[0]

                            # Add surv particle to lower border point near the previous part was removed
                            candidate=surv.loc[(surv['angle']<upper_ang) & (surv['angle']>lower_ang) & (surv['amplitude']<ploss['amplitude']),['id','angle','amplitude']]
                            if not candidate.empty:
                                border_min=pd.concat([ border_min,candidate.loc[[candidate.idxmax()["amplitude"]],:]])

                    border_min_fit=fit_DA(border_min['angle'], border_min['amplitude'], ang_range)
                    pb_border_max=border_max.loc[border_max['amplitude']<=border_min_fit(border_max['angle']),:]

                # Smooth DA
                if smoothing:
                    border_min,border_max=_da_smoothing(dd,border_min,border_max,at_turn=at_turn)
        
            # Save and return DA
            if self.meta.nseeds==0:
                row=f't{at_turn}'
            else:
                row=f't{at_turn} s{ss}'
#             self._set_da_and_border(ss,at_turn,border_min,border_max,ang_range)

        self._da     = pd.concat([self._da[    mask_da    ], *list_new_da    ], ignore_index=True)
        self._border = pd.concat([self._border[mask_border], *list_new_border], ignore_index=True)
#             new_da=pd.DataFrame({'DA lower':    [compute_da_1D(border_min['angle'], border_min['amplitude'], ang_range)],
#                                  'DAmin lower': [min(border_min['amplitude'])],
#                                  'DAmax lower': [max(border_min['amplitude'])],
#                                  'DA upper':    [compute_da_1D(border_max['angle'], border_max['amplitude'], ang_range)],
#                                  'DAmin upper': [min(border_max['amplitude'])],
#                                  'DAmax upper': [max(border_max['amplitude'])]
#                                 })
#             self._da=concat(self._da,new_da,seed=ss,t=at_turn)

# #             border_min['seed']=ss; border_min['t']=at_turn;
#             self._border=concat(self._border,
#                                 border_min.rename(columns={'angle':'ang lower', 'amplitude':'amp lower', 'id':'id lower'}, 
#                                                   inplace=False),
#                                 seed=ss,t=at_turn)

# #             border_max['seed']=ss; border_max['t']=at_turn;
#             self._border=concat(self._border,
#                                 border_max.rename(columns={'angle':'ang upper', 'amplitude':'amp upper', 'id':'id upper'}, 
#                                                   inplace=False),
#                                 seed=ss,t=at_turn)

#             self._da.loc[row,'seed']= ss
#             self._da.loc[row,'t']   = at_turn
#             self._da.loc[row,'DA lower']   = compute_da_1D(border_min['angle'], border_min['amplitude'],ang_range)
#             self._da.loc[row,'DAmin lower']= min(border_min['amplitude'])
#             self._da.loc[row,'DAmax lower']= max(border_min['amplitude'])
#             self._da.loc[row,'DA upper']   = compute_da_1D(border_max['angle'], border_max['amplitude'],ang_range)
#             self._da.loc[row,'DAmin upper']= min(border_max['amplitude'])
#             self._da.loc[row,'DAmax upper']= max(border_max['amplitude'])

#             if self.meta.nseeds==0:
#                 self._lower_davsturns.loc[at_turn,'turn'  ]=at_turn
#                 self._lower_davsturns.loc[at_turn,'border']=[ border_min ]
#                 self._lower_davsturns.loc[at_turn,'avg'   ]=compute_da_1D(border_min['angle'], border_min['amplitude'],ang_range)
#                 self._lower_davsturns.loc[at_turn,'min'   ]=min(border_min['amplitude'])
#                 self._lower_davsturns.loc[at_turn,'max'   ]=max(border_min['amplitude'])

#                 self._upper_davsturns.loc[at_turn,'turn'  ]=at_turn
#                 self._upper_davsturns.loc[at_turn,'border']=[ border_max ]
#                 self._upper_davsturns.loc[at_turn,'avg'   ]=compute_da_1D(border_max['angle'], border_max['amplitude'],ang_range)
#                 self._upper_davsturns.loc[at_turn,'min'   ]=min(border_max['amplitude'])
#                 self._upper_davsturns.loc[at_turn,'max'   ]=max(border_max['amplitude'])
#             else:

#                 self._lower_davsturns[ss].loc[at_turn,'turn'  ]=at_turn
#                 self._lower_davsturns[ss].loc[at_turn,'border']=[ border_min ]
#                 self._lower_davsturns[ss].loc[at_turn,'avg'   ]=compute_da_1D(border_min['angle'],border_min['amplitude'],ang_range)
#                 self._lower_davsturns[ss].loc[at_turn,'min'   ]=min(border_min['amplitude'])
#                 self._lower_davsturns[ss].loc[at_turn,'max'   ]=max(border_min['amplitude'])

#                 self._upper_davsturns[ss].loc[at_turn,'turn'  ]=at_turn
#                 self._upper_davsturns[ss].loc[at_turn,'border']=[ border_max ]
#                 self._upper_davsturns[ss].loc[at_turn,'avg'   ]=compute_da_1D(border_max['angle'],border_max['amplitude'],ang_range)
#                 self._upper_davsturns[seed].loc[at_turn,'min'   ]=min(border_max['amplitude'])
#                 self._upper_davsturns[seed].loc[at_turn,'max'   ]=max(border_max['amplitude'])
        sys.stdout.write(f'Computing DA at turn {at_turn} succesfully end!\n')

        if save:
            self.write_da()
            self.write_border()



    # Not allowed on parallel process
    # TODO: Fix description to fit the new data format
    def calculate_davsturns(self, from_turn: int=1, to_turn: Union[int,None]=None, bin_size: int=1, 
                            interp_order: str='1D', interp_method: str='trapz', force: bool=False, save: bool=True):
        """Compute the DA upper and lower evolution from a specific turn to another in the form of a pandas table ['turn','border','avg','min','max'] or for multiseeds { seed:['turn','border','avg','min','max'],  'stat':['turn','avg','min','max'] }

        Args:
            from_turn (int, optional):      First turn at which this estimation must be computed. Defaults to 1.
            to_turn (int | None, optional): Last turn at which this estimation must be computed. Defaults to max_turns.
            bin_size (int, optional):       The turns is slice by this number. Defaults to 1.
            interp_order (str, optional):   Interpolation order for the DA average: '1D', '2D', '4D'. Defaults to '1D'.
            interp_method (str, optional):  Interpolation method for the DA average: 'trapz', 'simpson', 'alternative_simpson'. Defaults to 'trapz'.
            force (bool, optional):         Flag reseting the da analysis. Defaults to False.
            save (bool, optional):          Flag for the saving da and border DataFrames in files. Defaults to True.
        """

        # Initialize input and da array
        if to_turn is None:
            to_turn=self.max_turns
        if to_turn > self.max_turns:
            raise ValueError(f'at_turn cannot be higher than the max number of turns for the simulation, here max_turn={DA.max_turns}')
            
        if interp_order=='1D':
            compute_da=compute_da_1D
        elif interp_order=='2D':
            compute_da=compute_da_2D
        elif interp_order=='4D':
            compute_da=compute_da_4D
        else:
            raise ValueError("interp_order must be either: '1D', '2D' or '4D'!")

        if interp_method=='trapz':
            interp=trapz
        elif interp_method=='simpson':
            interp=simpson
        elif interp_method=='alternative_simpson':
            interp=alter_simpson
        else:
            raise ValueError("interp_method must be either: 'trapz', 'simpson', 'alternative_simpson'!")
        ang_range=(self.meta.ang_min,self.meta.ang_max)

#         if self._lower_davsturns is None or (self.meta.nseeds==0 and to_turn not in self._lower_davsturns)or  (self.meta.nseeds>0 and to_turn not in self._lower_davsturns[1]):
        if force:
            self._da = None
            self._border = None
        elif self._da is None or self._border is None:
            self.read_da()
            self.read_border()
        if self._da is None or to_turn not in np.unique(self._da['t']):
            self.calculate_da(at_turn=to_turn, smoothing=True, 
                              interp_order=interp_order, interp_method=interp_method, force=force, save=False) 


        # Load particle data
        data = self.survival_data.copy()
        data['id']= data.index
        
        # Run DA raw border detection
        if self.da_type not in ['monte_carlo', 'free']:
            # Create the round_angle columns
            data['round_angle']= data['angle']

#             import time
#             ti = time.time()

            # Get list of turn to 
            lturns=np.sort(np.unique(bin_size*np.floor(self.survival_data.nturns/bin_size)))
            lturns=lturns[(lturns>from_turn) & (lturns<to_turn)]
            lturns=np.unique([from_turn, *lturns, to_turn]).astype(int)

            # Remove turns already computed
            if self.meta.nseeds==0:
                lturns=np.sort(list(set(lturns) - set(self.t_steps)))
                list_new_da     = np.array([None for ii in range(len(lturns))])
                list_new_border = np.array([None for ii in range(len(lturns))])
            else:
                lturns=np.sort(list(set(lturns) - set(np.concatenate(list(self.t_steps.values())))))
                list_new_da     = np.array([None for ii in range(len(lturns)*self.meta.nseeds)])
                list_new_border = np.array([None for ii in range(len(lturns)*self.meta.nseeds)])
            mask_da     = [True] * len(self._da)
            mask_border = [True] * len(self._border)

#             te = time.time(); print(f"calculate_davsturns: pre loop {te-ti:.4}s") ; ti = te;

            # Loop in turn to compute the da
            idx = 0
            for at_turn in reversed(lturns):
                # Select the list of seed to comput at this turn
#                 if self.meta.nseeds==0 or at_turn==from_turn or at_turn==to_turn or bin_size>1:
#                     list_seed=None
#                 else:
#                     list_seed=np.sort(np.unique(self.survival_data.loc[self.survival_data.nturns==at_turn,'seed']))

                if self.meta.nseeds==0:
                    list_seed=[ 0 ]
                    list_data=[ data ]
                elif at_turn==from_turn or at_turn==to_turn or bin_size>1:
                    list_seed=[s for s in range(1,self.meta.nseeds+1)]
                    list_data=[ data.loc[data.seed==s,:] for s in list_seed ]
                else:
                    list_seed=np.sort(np.unique(data.loc[data.nturns==at_turn,'seed']))
                    list_data=[ data.loc[data.seed==s,:] for s in list_seed ]
                    
#                 te = time.time(); print(f"calculate_davsturns: select in loop1 {te-ti:.4}s") ; ti = te;

#                 self.calculate_da(at_turn=at_turn,angular_precision=1,smoothing=False,list_seed=list_seed)

                # Run DA raw border detection
#                 sys.stdout.write(f'seed {0:>3d}')
                for ss,dd in zip(list_seed,list_data):
#                     sys.stdout.write(f'\rseed {seed:>3d}')

                    # Get DA raw estimation
#                     border_min, border_max =_da_raw(dd,at_turn,seed,compute_da, ang_range)
                    list_new_da[idx], list_new_border[idx] =_da_raw(dd,at_turn,ss,compute_da, ang_range)
#                     te = time.time(); print(f"calculate_davsturns: da_raw in loop2 {te-ti:.4}s") ; ti = te;

#                     self._set_da_and_border(ss,at_turn,border_min,border_max,ang_range)
#                     te = time.time(); print(f"calculate_davsturns: set_da in loop2 {te-ti:.4}s") ; ti = te;
                    
                    mask_da     = mask_da     & ~( ( self._da.seed    ==ss ) & ( self._da.t    ==at_turn ) )
                    mask_border = mask_border & ~( ( self._border.seed==ss ) & ( self._border.t==at_turn ) )
                    idx += 1
                sys.stdout.write(f'Computing DA at turn {at_turn} succesfully end!\n')
            self._da     = pd.concat([self._da[    mask_da    ], *list_new_da    ], ignore_index=True)
            self._border = pd.concat([self._border[mask_border], *list_new_border], ignore_index=True)
#                 sys.stdout.write(f'\r')
        else:
            # Transform data from x-y format to ang-amp
            if 'angle' not in data.columns or 'amplitude' not in data.columns:
                data['angle']      = np.arctan2(data['y'],data['x'])*180/np.pi
                data['amplitude']  = np.sqrt(data['x']**2+data['y']**2)
                
            if self.meta.nseeds==0:
                list_seed=[ 0 ]
                list_data=[ data ]
            else:
                list_seed=[ ss for ss in range(1,self.meta.nseeds+1) ]
                list_data=[ data[data.seed==ss].copy() for ss in range(1,self.meta.nseeds+1) ]
            
            for seed,data in zip(list_seed,list_data):
#                 if self.meta.nseeds==0:
#                     lower_davsturns=self._lower_davsturns
#                     upper_davsturns=self._upper_davsturns
#                 else:
#                     lower_davsturns=self._lower_davsturns[seed+1]
#                     upper_davsturns=self._upper_davsturns[seed+1]
#                 ang_range=(min(data['angle']),max(data['angle']))

                # Get list of turn to 
                t_steps=self.t_steps
                lturns=np.sort(np.unique(bin_size*np.floor(data.nturns/bin_size)))
                lturns=lturns[(lturns>from_turn) & (lturns<to_turn)]
                lturns=np.unique(np.append(lturns,[from_turn,to_turn])).astype(int)
#                 lturns=np.array([at_turn for at_turn in lturns if at_turn not in lower_davsturns])
                if self.meta.nseeds==0:
                    lturns=np.array([at_turn for at_turn in lturns if at_turn not in t_steps])
                else:
                    lturns=np.array([at_turn for at_turn in lturns if any([at_turn not in t_steps[seed] for seed in range(1,self.meta.nseeds+1)]) ])


                upper_notmonotonius=False
#                 border_min=lower_davsturns.loc[to_turn,'border'][0]
#                 DA_lim_max=max(upper_davsturns.loc[to_turn,'border'][0]['amplitude'])
#                 da=lower_davsturns.loc[to_turn,'avg']
                border_min=get_border_min(self,seed,to_turn)
                border_max=get_border_max(self,seed,to_turn)
                DA_lim_max=max(border_max['amplitude'])
                da=select_rows(self._da,seed=seed,t=to_turn)['DA lower']
                removed=pd.DataFrame(columns=data.columns)
                for at_turn in reversed(lturns):
                    # Initiallise loop
                    raw_border_min=border_min

                    # Regenerate the upper da border and detect lower da limit
                    losses=data.nturns<at_turn
                    loss=data.loc[ losses,:]; min_loss=min(loss['amplitude'])
                    surv=data.loc[~losses,:]; max_surv=max(surv['amplitude'])
                    max_amplitude = max([min_loss,max_surv])+2
                    raw_border_max=loss.loc[loss['amplitude']<=max_amplitude,['id','angle','amplitude']]

                    # Check if DA border cross each others
                    fit_max=fit_DA(raw_border_max['angle'], raw_border_max['amplitude'], ang_range)
                    check_boundary_cross=raw_border_min.loc[raw_border_min['amplitude']>=fit_max(raw_border_min['angle'])]

                    # Recheck previous turns
                    if not check_boundary_cross.empty and self.da_type in ['monte_carlo', 'free']:
                        removed=pd.concat([ removed, data.loc[check_boundary_cross.id,:] ])
                        recheck=np.sort(np.unique(data.nturns))
                        recheck=recheck[recheck>at_turn]
                        new_DA_lim_min=0; da=0
                        for rc in reversed(recheck):
                            surv_rc=data.loc[data.nturns>=rc,:]
#                             raw_border_min=lower_davsturns.loc[rc,'border'][0]
#                             raw_border_max=upper_davsturns.loc[rc,'border'][0]
                            raw_border_min=get_border_min(self,seed,rc)
                            raw_border_max=get_border_max(self,seed,rc)
                            raw_border_min=raw_border_min.loc[raw_border_min['amplitude']<fit_max(raw_border_min['angle']),:]
                            new_border_min,new_border_max=_da_smoothing(data,raw_border_min,raw_border_max,removed=removed,
                                                                        at_turn=rc,DA_lim_min=new_DA_lim_min,
                                                                        active_warmup=False)

                            new_da=compute_da_1D(new_border_min['angle'],new_border_min['amplitude'],ang_range)
                            new_DA_lim_min=min(new_border_min['amplitude'])

                            # Check if DA decrease with the turns
                            if new_da>=da:
                                da=new_da
                                border_min=new_border_min;

                            # Save DA
                            self._set_da_and_border(seed,rc,border_min,border_max,ang_range)


#                             lower_davsturns.loc[rc,'border']=[ border_min ]
#                             lower_davsturns.loc[rc,'avg'   ]=compute_da(border_min['angle'],
#                                                                         border_min['amplitude'],ang_range,interp)
#                             lower_davsturns.loc[rc,'min'   ]=new_DA_lim_min
#                             lower_davsturns.loc[rc,'max'   ]=max(border_min['amplitude'])

#                             upper_davsturns.loc[rc,'border']=[ new_border_max ]
#                             upper_davsturns.loc[rc,'avg'   ]=compute_da(new_border_max['angle'],
#                                                                         new_border_max['amplitude'],ang_range,interp)
#                             upper_davsturns.loc[rc,'min'   ]=min(new_border_max['amplitude'])
#                             upper_davsturns.loc[rc,'max'   ]=max(new_border_max['amplitude'])

                        raw_border_min=new_border_min
                        raw_border_max=loss.loc[loss['amplitude']<=max_amplitude,['id','angle','amplitude']]

                    # Add new surviving particles to lower da border and smooth da borders
                    DA_lim_min=min(raw_border_min['amplitude'])
                    new_border_min,new_border_max=_da_smoothing(data,raw_border_min,raw_border_max,removed=removed,
                                                                at_turn=at_turn,DA_lim_min=DA_lim_min,active_warmup=True)
                    new_da=compute_da_1D(new_border_min['angle'], new_border_min['amplitude'],ang_range)

                    # Check if DA decrease with the turns
                    if max(new_border_max['amplitude'])<DA_lim_max:
                        upper_notmonotonius=True
#                         print(f'Warning: Detect upper border limit problem ({at_turn=})')

                    # Check if DA decrease with the turns
                    if new_da>=da:
                        da=new_da
                        border_min=new_border_min
                    border_max=new_border_max

                    # Save DA
                    self._set_da_and_border(seed,at_turn,border_min,border_max,ang_range)

#                     lower_davsturns.loc[at_turn,'turn'  ]=at_turn
#                     lower_davsturns.loc[at_turn,'border']=[ border_min ]
#                     lower_davsturns.loc[at_turn,'avg'   ]=compute_da(border_min['angle'],border_min['amplitude'],
#                                                                      ang_range,interp)
#                     lower_davsturns.loc[at_turn,'min'   ]=min(border_min['amplitude'])
# #                     lower_davsturns.loc[at_turn,'min'   ]=new_DA_lim_min=min(border_min['amplitude'])
#                     lower_davsturns.loc[at_turn,'max'   ]=max(border_min['amplitude'])

#                     upper_davsturns.loc[at_turn,'turn'  ]=at_turn
#                     upper_davsturns.loc[at_turn,'border']=[ border_max ]
#                     upper_davsturns.loc[at_turn,'avg'   ]=compute_da(border_max['angle'],border_max['amplitude'],
#                                                                      ang_range,interp)
#                     upper_davsturns.loc[at_turn,'min'   ]=min(border_max['amplitude'])
#                     upper_davsturns.loc[at_turn,'max'   ]=DA_lim_max=max(border_max['amplitude'])

                if upper_notmonotonius:
                    print("#TODO: Rework the monoticity routine")
                    #TODO: Rework the monoticity routine
#                     # Impose Monotonicity of the upper max value
#                     prev_DA_lim_min=lower_davsturns.loc[min(lturns),'min']
#                     prev_DA_lim_max=upper_davsturns.loc[min(lturns),'max']
#                     for idx in range(len(lturns)):
#                         at_turn=lturns[idx]
#                         DA_lim_max=prev_DA_lim_max
#                         raw_DA_lim_max=upper_davsturns.loc[at_turn,'max']

#                         if raw_DA_lim_max>DA_lim_max:
#                             raw_DA_lim_min=lower_davsturns.loc[at_turn,'min']
#                             raw_border_min=lower_davsturns.loc[at_turn,'border'][0]
#                             raw_border_max=upper_davsturns.loc[at_turn,'border'][0]

#                             # Remove border particles higher than the upper limit
#                             raw_border_max=raw_border_max.loc[raw_border_max['amplitude']<=DA_lim_max]
#                             fit_max=fit_DA(raw_border_max['angle'], raw_border_max['amplitude'], ang_range)
#                             raw_border_min=raw_border_min.loc[raw_border_min['amplitude']<=fit_max(raw_border_min['angle'])]

#                             # Add new surviving particles to lower da border and smooth da borders
#                             DA_lim_min=min(raw_border_min['amplitude'])
#                             new_border_min,new_border_max=_da_smoothing(data,raw_border_min,raw_border_max,
#                                                                         at_turn=at_turn,removed=removed,
#                                                                         DA_lim_min=DA_lim_min,DA_lim_max=DA_lim_max,
#                                                                         active_warmup=True)

#                             # Save DA
#                             self._set_da_and_border(seed,at_turn,new_border_min,new_border_max,ang_range)

#                             lower_davsturns.loc[at_turn,'border']=[ new_border_min ]
#                             lower_davsturns.loc[at_turn,'avg'   ]=compute_da(new_border_min['angle'],
#                                                                              new_border_min['amplitude'],ang_range,interp)
#                             lower_davsturns.loc[at_turn,'min'   ]=min(new_border_min['amplitude'])
# #                             lower_davsturns.loc[at_turn,'min'   ]=new_DA_lim_min=min(new_border_min['amplitude'])
#                             lower_davsturns.loc[at_turn,'max'   ]=max(new_border_min['amplitude'])

#                             upper_davsturns.loc[at_turn,'border']=[ new_border_max ]
#                             upper_davsturns.loc[at_turn,'avg'   ]=compute_da(new_border_max['angle'],
#                                                                              new_border_max['amplitude'],ang_range,interp)
#                             upper_davsturns.loc[at_turn,'min'   ]=min(new_border_max['amplitude'])
#                             upper_davsturns.loc[at_turn,'max'   ]=max(new_border_max['amplitude'])

#                             prev_DA_lim_min=min(new_border_min['amplitude'])
#                             prev_DA_lim_max=max(new_border_max['amplitude'])

#                             # Recheck previous turns for non-monoticity of the lower DA estimation
#                             DA_lim_min=prev_DA_lim_min
#                             border_min=new_border_min
#                             if any(lower_davsturns.loc[lturns[:idx-1],'min']<DA_lim_min):
#                                 for rc in reversed(lturns[:idx-1]):
#                                     if lower_davsturns.loc[rc,'min']<DA_lim_min:
#                                         raw_border_min=border_min #lower_davsturns.loc[rc,'border'][0]
#                                         raw_border_max=upper_davsturns.loc[rc,'border'][0]
#                                         DA_lim_max=upper_davsturns.loc[rc,'max']


#                                         new_border_min,new_border_max=_da_smoothing(data,raw_border_min,raw_border_max,
#                                                                                     at_turn=rc,removed=removed,
#                                                                                     DA_lim_min=DA_lim_min,DA_lim_max=DA_lim_max,
#                                                                                     active_warmup=False)


#                                         # Check if DA decrease with the turns
#                                         new_da_min=compute_da_1D(new_border_min['angle'],new_border_min['amplitude'],ang_range)
#                                         raw_da_min=compute_da_1D(border_min['angle'],border_min['amplitude'])
#                                         if new_da_min<raw_da_min:
#                                             print(1)
#                                             new_border_min=border_min
#                                             new_border_max=upper_davsturns.loc[rc,'border'][0]

#                                         # Save DA
#                                         self._set_da_and_border(seed,rc,new_border_min,new_border_max,ang_range)

#                                         lower_davsturns.loc[rc,'border']=[ new_border_min ]
#                                         lower_davsturns.loc[rc,'avg'   ]=compute_da(new_border_min['angle'],
#                                                                                     new_border_min['amplitude'],
#                                                                                     ang_range,interp)
#                                         lower_davsturns.loc[rc,'min'   ]=min(new_border_min['amplitude'])
#                                         lower_davsturns.loc[rc,'max'   ]=max(new_border_min['amplitude'])

#                                         upper_davsturns.loc[rc,'border']=[ new_border_max ]
#                                         upper_davsturns.loc[rc,'avg'   ]=compute_da(new_border_max['angle'],
#                                                                                     new_border_max['amplitude'],
#                                                                                     ang_range,interp)
#                                         upper_davsturns.loc[rc,'min'   ]=min(new_border_max['amplitude'])
#                                         upper_davsturns.loc[rc,'max'   ]=max(new_border_max['amplitude'])

#                                         border_min=new_border_min
#                                         DA_lim_min=lower_davsturns.loc[rc,'min']


#                         else:
#                             prev_DA_lim_max=raw_DA_lim_max


#                 if self.meta.nseeds==0:
#                     self._lower_davsturns=lower_davsturns
#                     self._upper_davsturns=upper_davsturns
#                 else:
#                     self._lower_davsturns[seed]=lower_davsturns
#                     self._upper_davsturns[seed]=upper_davsturns

        # For the multiseeds case, generate the summary as 'stat' over all the seeds
        if self.meta.nseeds!=0:
            print("#TODO: Rework stat routine")
            #TODO: Rework stat routine
#             stat_lower_davsturns=pd.DataFrame({},index=lturns,columns=['turn','border','avg','min','max'])
#             stat_upper_davsturns=pd.DataFrame({},index=lturns,columns=['turn','border','avg','min','max'])

#             # Compute the stat
#             lower_davsturns=pd.DataFrame({},index=[s for s in range(1,self.meta.nseeds+1)], columns=['avg','min','max']) 
#             upper_davsturns=pd.DataFrame({},index=[s for s in range(1,self.meta.nseeds+1)], columns=['avg','min','max'])
#             sys.stdout.write(f'Compute turn-by-turn statistic... (turn={int(lturns[0]):>7d}, seed={1:>3d})') 
#             for at_turn in lturns:
#                 for s in range(1,self.meta.nseeds+1):
#                     sys.stdout.write(f'\rCompute turn-by-turn statistic... (turn={int(at_turn):>7d}, seed={s:>3d})')
#                     DA=self.get_lower_da(at_turn=at_turn,seed=s)
#                     lower_davsturns.loc[s,'avg']=DA['avg']
#                     lower_davsturns.loc[s,'min']=DA['min']
#                     lower_davsturns.loc[s,'max']=DA['max']

#                     DA=self.get_upper_da(at_turn=at_turn,seed=s)
#                     upper_davsturns.loc[s,'avg']=DA['avg']
#                     upper_davsturns.loc[s,'min']=DA['min']
#                     upper_davsturns.loc[s,'max']=DA['max']

#                 # Save stat
#                 stat_lower_davsturns.loc[at_turn,'turn']=at_turn
#                 stat_lower_davsturns.loc[at_turn,'avg']=lower_davsturns['avg'].mean()
# #                 stat_lower_davsturns.loc[at_turn,'min']=lower_davsturns['min'].min()
# #                 stat_lower_davsturns.loc[at_turn,'max']=lower_davsturns['max'].max()
#                 stat_lower_davsturns.loc[at_turn,'min']=lower_davsturns['avg'].min()
#                 stat_lower_davsturns.loc[at_turn,'max']=lower_davsturns['avg'].max()

#                 stat_upper_davsturns.loc[at_turn,'turn']=at_turn
#                 stat_upper_davsturns.loc[at_turn,'avg']=upper_davsturns['avg'].mean()
# #                 stat_upper_davsturns.loc[at_turn,'min']=upper_davsturns['min'].min()
# #                 stat_upper_davsturns.loc[at_turn,'max']=upper_davsturns['max'].max()
#                 stat_upper_davsturns.loc[at_turn,'min']=upper_davsturns['avg'].min()
#                 stat_upper_davsturns.loc[at_turn,'max']=upper_davsturns['avg'].max()
#             sys.stdout.write(f'\rCompute turn-by-turn statistic... Done!\n')
#             self._lower_davsturns['stat']=stat_lower_davsturns
#             self._upper_davsturns['stat']=stat_upper_davsturns
        if save:
            self.write_da()
            self.write_border()

    # =================================================================
    # ========================== Fit models ===========================
    # =================================================================

    # Not allowed on parallel process
    def _fit_model(self, nb_param: int, data_type: tuple[str,str], model, name: str='user', 
                   model_default: dict={}, model_boundary: dict={}, model_mask=None, seed: Union[int,None]=None, 
                   nrand: int=1000, nsig: int=2, ntry: int=2, save: bool=True, force: bool=False, fix_seed: bool=True):
        """DA vs turns fitting procedure.

        Args:
            nb_param (int):                         Number of parameter from the Model used.
            data_type (tuple[str,str]):             Which data is used as a tuplet: (type1,type2) with type1 in ['lower','upper','uniform','normal'] and type2 in ['min','max','avg'].
            model (list | function):                Either an element from ['2','2b','2n','4','4b','4n'] or a function. In the later case, also give model_default and model_boundary.
            name (str, optional):                   Name of the model in the dataframe if not a builtin one. Defaults to 'user'.
            model_default (dict, optional):         A dict of the model parameters default values with parameters name as keys. Defaults to {}.
            model_boundary (dict, optional):        A dict of the model parameters boundarie with parameters name as keys. Defaults to {}.
            model_mask (function | None, optional): Mask function filtering the allowed data for the fitting procedure. Defaults to None.
            seed (int | None, optional):            The seed number for the multisees case. Defaults to None.
            nrand (int, optional):                  Number of points/try for the normal and uniform fitting. Defaults to 1000.
            nsig (int, optional):                   Scaling of distribution for the normal and uniform fitting. Defaults to 2.
            ntry (int, optional):                   Number of try for the normal and uniform fitting. Defaults to 2.
            save (bool, optional):                  Flag to save the results. Defaults to True.
            force (bool, optional):                 Flag to erase previous results. Defaults to False.
            fix_seed (bool, optional):              Flag to fix the seed generation normal and uniform fitting.  Defaults to True.
        """
        
        if self._da_model is None:
            self.read_da_model()

        # Model selection
        name, model, mdefault, mboundary, mmask=select_model(model,model_default=model_default,model_boundary=model_boundary,
                                                             model_mask=model_mask,name=name)
        if 'N0' in mboundary:
            mboundary['N0'][1]=self.max_turns

        # Data type selection
        if self.meta.nseeds==0:
            seed=0
            row=pd.MultiIndex.from_tuples([(data_type[0],data_type[1])], names=["method", "type"])
        if self.meta.nseeds!=0:
            row=pd.MultiIndex.from_tuples([(data_type[0],data_type[1],f'{seed}')], names=["method", "type", "seed"])
        davsturn=select_rows(self._da,seed=seed)
        
        if fix_seed and data_type[0] in ['uniform','normal']:
            print("# Fix seed for model fitting")
            np.random.seed(0)

        if data_type[1]=='avg':
            dtemp='DA'
        elif data_type[1] in ['min','max']:
            dtemp='DA'+data_type[1]
        else:
            dtemp=data_type[1]
        xdata =np.array(davsturn.loc[:,'t'].values, dtype=float)
        ylower=np.array(davsturn.loc[:, dtemp+' lower' ].values, dtype=float)
        yupper=np.array(davsturn.loc[:, dtemp+' upper' ].values, dtype=float)

        if 'lower' == data_type[0]:
            lx=[xdata]; ly=[ylower]
        elif 'upper' == data_type[0]:
            lx=[xdata]; ly=[yupper]
        elif 'average' == data_type[0]:
            lx=[xdata]; ly=[0.5*(yupper+ylower)]
        elif data_type[0] in ['uniform','normal']:
            row_std=pd.MultiIndex.from_tuples([(data_type[0][:4]+'std',data_type[1],f'{seed}')], 
                                              names=["method", "type", "seed"])
            rturns=(min(xdata),max(xdata))

            lx=np.empty([ntry,nrand]); ly=np.empty([ntry,nrand])
            for ii in range(ntry):
                xrand=np.floor(10**( (np.log10(rturns[1])-np.log10(rturns[0]))*np.random.uniform(size=[nrand])+np.log10(rturns[0]) )).astype(int)
                if 'uniform' == data_type[0]:
                    yrand=np.random.uniform(size=[nrand])
                    for tmax,tmin,ylo,yup in zip(xdata[0:-1],xdata[1:],ylower[1:],yupper[1:]):
                        mask= (xrand>=tmin) & (xrand<tmax)
                        yrand[mask]=(yup-ylo)*yrand[mask]+ylo

                elif 'normal' == data_type[0]:
                    yrand=np.random.normal(loc=0.0,scale=0.5,size=[nrand])
                    for tmax,tmin,ylo,yup in zip(xdata[0:-1],xdata[1:],ylower[1:],yupper[1:]):
                        mask= (xrand>=tmin) & (xrand<tmax)
                        yrand[mask]=(yup-ylo)*(yrand[mask]/nsig+0.5)+ylo
                x=xrand; y=yrand
                lx[ii,:]=x; ly[ii,:]=y

        else:
            raise ValueError(f'data_type must be one of the following form: ([lower,upper,uniform,normal],[min,max,avg]\n   -> input {data_type}')

        if len(ly[0])<=nb_param:
            raise ValueError('There is not enougth data for the fitting procedure.')

        # Manage duplicate analysis
        if not force and self._da_model is not None:
            # Multicolumns
            col=f'{name} {nb_param} res'
            if self._da_model.index.isin([row[-1]]).any() and any(self._da_model.columns==col) and \
            not np.isnan(self._da_model.loc[row[-1],col]) and self._da_model.loc[row[-1],col]>0:
                return
        if self.meta.nseeds!=0:
            print(f'Running {data_type} Model {name} (Nb. param: {nb_param}) for seed {seed}')
        else:
            print(f'Running {data_type} Model {name} (Nb. param: {nb_param})')

        # Fit the model
        keys=np.array([k for k in mboundary.keys()])
        res=pd.DataFrame()
        for s,(x,y) in enumerate(zip(lx,ly)):
            sres={}
            smdefault=mdefault.copy()
            for nprm in range(1,min(nb_param,len(keys))+1):
                # Select param default and boundary values
                keys_fit=keys[:nprm]
                dflt=tuple([smdefault[k]  for k in keys_fit])
                bndr=([mboundary[k][0] for k in keys_fit],[mboundary[k][1] for k in keys_fit])

                try:
                    # Fit model to plots
                    dflt_new, sg=curve_fit(model,x,y,p0=dflt,bounds=bndr)
#                     print(f"{smdefault=}")
#                     tmp={kk:vv for kk,vv in zip(keys_fit,dflt_new)}
#                     print(f"{tmp=}")
                    error=((y - model(x,**{kk:vv for kk,vv in zip(keys_fit,dflt_new)}))**2).sum() / (len(y)-nprm)
#                     error=((y - model(x,**smdefault))**2).sum() / (len(y))
                except Exception as err:
                    warnings.warn(f"[{type(err)}] No solution found for {data_type} Model {name} (Nb. param: {nprm}). Return -1.")
                    dflt_new=tuple([-1 for k in dflt]); error=-1

                # Select param default and boundary values
                for k in range(0,nprm):
                    smdefault[keys_fit[k]]=dflt_new[k]
                    sres[f'{name} {nprm} {keys_fit[k]}']= [ dflt_new[k] ]
                sres[f'{name} {nprm} res']= [ error ]
            if res.empty:
                res=pd.DataFrame(sres)
            else:
                res=pd.concat([res,pd.DataFrame(sres)], ignore_index=True)

        # Prepare results to be saved
        if data_type[0] in ['uniform','normal']:
            res_avg=pd.DataFrame(res.mean()).T; res_avg.index=row
            res_std=pd.DataFrame(res.std()).T; res_std.index=row_std
            res=pd.concat([res_avg,res_std])
            res[f'seed']= seed
            res.loc[row    ,f'fit']= dtemp+' '+data_type[0]
            res.loc[row_std,f'fit']= dtemp+' '+data_type[0][:4]+'std'
        else:
            res.index=row
            res[f'seed']= seed
            res[f'fit']= dtemp+' '+data_type[0]

        # Save results
        self._da_model=concat(self._da_model,res.loc[row,:],seed=seed,fit=dtemp+' '+data_type[0])
        if data_type[0] in ['uniform','normal']:
            self._da_model=concat(self._da_model,res.loc[row_std,:],seed=seed,fit=dtemp+' '+data_type[0][:4]+'std')
            
        if save:
            self._da_model=self._da_model.fillna(-1)
            self.write_da_model()

            
    # Not allowed on parallel process
    def _fit_model_from_list(self, nb_param: int, list_data_types: Union[list,None]=None, list_models: list=['2','2b','2n','4','4b','4n'],
                             list_seeds: Union[list,None]=None, **kwargs):
                            # list_seeds: Union[list,None]=None, nrand: int=1000, nsig: int=2, ntry: int=1, force: bool=False):
        """DA vs turns fitting procedure for a list of data_types, model or seed.

        Args:
            nb_param (int):                               Number of parameter from the Model used.
            list_data_types (list | None, optional): List of data types as defined in `_fit_model`. Defaults to None.
            list_models (list, optional):                 List of in-build model as defined in `_fit_model`. Defaults to ['2','2b','2n','4','4b','4n'].
            list_seeds (list | None, optional):      List of seeds for the multiseed case. Defaults to None.
        """
        if list_data_types is None:
#             list_data_types=[f'{d1}_{d2}' for d1 in ['lower','upper','uniform','normal'] for d2 in ['min','avg','max']]
            list_data_types=[(d1,d2) for d1 in ['lower','upper','uniform','normal'] for d2 in ['min','avg','max']]
        if list_seeds is None:
            if self.meta.nseeds!=0:
                list_seeds = [ss for ss in range(1,self.meta.nseeds+1)].append('stat')
            else:
                list_seeds = [None]
        
        for dt in list_data_types:
            for md in list_models:
                for ss in list_seeds:
                    self._fit_model(nb_param=nb_param,data_type=dt,model=md,seed=ss,
                                    save=False,**kwargs)
                                    #nrand=nrand,ntry=ntry,nsig=nsig,save=False,force=force)
        
        self._da_model=self._da_model.fillna(-1)
        self.write_da_model()
        
        
    # Not allowed on parallel process
    def get_model_parameters(self, data_type: list[str], model, nb_parm: int,
                             keys: Union[list,None]=None, seed: Union[list,None]=None, name: str='user'): # -> tuple[function , dict, float]:
        """DA vs turns fitting procedure for a list of data_types, model or seed.
        Args:
            data_type (list[str]):        Data types as defined in `_fit_model`.
            model (str or func):          List of in-build model as defined in `_fit_model`. If it's not a build in model, please also give parameter name in keys.
            nb_parm (int):                Number of parameter from the Model used.
            keys (list | None, optional): List of parameters name. Defaults to None.
            seed (int, optional):         Seed number for the multiseed case. Defaults to None.
            name (str, optional):         Name of the model if nodel is a function. Defaults to 'user'.
            
        Returns:
            (func, dict, float):  The model as a `function`; a `dict` of parameters with their values after the fit; and the Residut from the fiting.
        
        """
        if self._da_model is None:
            self.read_da_model()
        if self._da_model is None:
            raise FileNotFoundError('No data for model fitting have been found.')

        # Model selection
        name, model, mdefault, dmp, dmp=select_model(model,name=name)
        if keys is None and mdefault=={}:
            raise ValueError('Please specify the parameters name as keys.')
        else:
            keys=[k for k in mdefault.keys()];

        if seed is None and self.meta.nseeds!=0:
            raise ValueError('Please specify the seed.')
        elif seed is None and self.meta.nseeds==0:
            seed=0

#         if self.meta.nseeds!=0:
#             row=pd.MultiIndex.from_tuples([(data_type[0],data_type[1],str(seed))], names=["method", "type", "seed"])
#         else:
#             row=pd.MultiIndex.from_tuples([(data_type[0],data_type[1])], names=["method", "type"])


        if data_type[1]=='avg':
            dtemp='DA'
        elif data_type[1] in ['min','max']:
            dtemp='DA'+data_type[1]
        else:
            dtemp=data_type[1]
        data=select_rows(self._da_model,seed=seed,fit=dtemp+' '+data_type[0])
        data.reset_index(inplace=True)
        # Multicolumns
    #         param={k:self._da_model.loc[row[0],(name,nb_parm,k)] for k in keys}
    #         return model, param, self._da_model.loc[row[0],(name,nb_parm,'res')]

    #     # String columns
    #     param={k:self._da_model.loc[row[0],f'{name} {nb_parm} {k}'] for k in keys[:nb_parm]}
    #     return model, param, self._da_model.loc[row[0],f'{name} {nb_parm} res']

        # New format
#         print(f"{self._da_model=}")
#         print(f"{dtemp+' '+data_type[0]=}")
#         print(f"{name=}")
#         print(f"{nb_parm=}")
#         print(f"{data=}")
        param={k:data.loc[0,f'{name} {nb_parm} {k}'] for k in keys[:nb_parm]}
        return model, param, data.loc[0,f'{name} {nb_parm} res']
            
    
                
    # =================================================================
    # ==================== Manage tracking jobs =======================
    # =================================================================

    # Allowed on parallel process
    def _create_job(self, npart: Union[int,None]=None, logging: bool=True, force_single_seed_per_job:Union[bool,None]=None):
        def _get_seeds_and_stuff(npart, logging):
            mask = (self._surv['submitted'] == False)
#             if npart is None:
#                 this_part_ids = self._surv[mask].index
#             else:
#                 this_part_ids = self._surv[mask].index[:npart]
#             if self.meta.nseeds > 0:
#                 df = self._surv.loc[this_part_ids]
#                 seeds = np.unique(df['seed'])
#                 if force_single_seed_per_job:
#                     # Only take jobs from one seed
#                     seeds = seeds[:1]
#                     mask = df['seed'] == seeds[0]
#                     this_part_ids = df.loc[mask].index
#             else:
#                 seeds = None
            # Select the seeds before such that if particles are not in the right order in
            # _surv, they are still taken into consideration (ex: pair particles)
            if any(mask):
                if self.meta.nseeds > 0:
    #                 df = self._surv.loc[mask]
                    seeds = np.unique(self._surv.loc[mask,'seed'])
                    if force_single_seed_per_job:
                        # Only take jobs from one seed
                        seeds = seeds[:1]
                        mask = (mask) & (self._surv['seed'] == seeds[0])
                else:
                    seeds = None
            else:
                seeds=-1
            if npart is None:
                this_part_ids = self._surv[mask].index
            else:
                this_part_ids = self._surv[mask].index[:npart]

            # Quit the job if no particles need to be submitted
            if len(this_part_ids) == 0:
                print("No more particles to submit! Exiting...")
                if logging:
                    # The submissions log for this job will only have a status field
                    self.meta.update_submissions(self._active_job, {'status': 'No submission needed.'})
                return this_part_ids, seeds, -1
            # Otherwise, flag the particles as submitted, before releasing the file again
            self._surv.loc[this_part_ids, 'submitted'] = True
            return this_part_ids, seeds, 0


        # Get job ID
        self._active_job = self.meta.new_submission_id() if logging else None

        # Initialise force_single_seed_per_job
        if npart is None:
            # If tracking all particles, default to all seeds as well
            if force_single_seed_per_job is None:
                force_single_seed_per_job = False
            txt = ' over all seeds' if self.meta.nseeds > 0 and not force_single_seed_per_job else ''
            print(f"Tracking all available particles{txt}. Make sure this is not ran on a parallel process, " + \
                   "as results might be unpredictable and probably wrong.")
        else:
            force_single_seed_per_job = True if force_single_seed_per_job is None else force_single_seed_per_job

        # Get available particles
        if self.surv_exists():
            with ProtectFile(self.meta.surv_file, 'r+b', wait=_db_access_wait_time) as pf:
                # Get the first npart particle IDs that are not yet submitted
                # TODO: this can probably be optimised by only reading last column
                self.read_surv(pf)
                this_part_ids, seeds, flag = _get_seeds_and_stuff(npart, logging)
                if flag == 0:
                    self.write_surv(pf)
        else:
            this_part_ids, seeds, flag = _get_seeds_and_stuff(npart, logging)

        # Prepare job
        if flag == 0:
            # Reduce dataframe to only those particles in this job
            self._surv = self._surv.loc[this_part_ids]

            # Submission info
            if logging:
                self._active_job_log = {
                        'submission_time': datetime.datetime.now().isoformat(),
                        'finished_time':   0,
                        'status':          'Running',
                        'tracking_turns':  self.meta.max_turns,
                        'particle_ids':    '[' + ', '.join([str(pid) for pid in this_part_ids]) + ']',
                        'seeds':           [ int(seed) for seed in seeds ] if seeds is not None else None,
                        'warnings':        [],
                        'output':          [],
                }

        return this_part_ids, seeds, flag



    # Allowed on parallel process
    def _update_job_log(self, update, logging=True):
        if logging:
            self._active_job_log.update(update)
            self.meta.update_submissions(self._active_job, self._active_job_log)

    # Allowed on parallel process
    def _append_job_log(self, key, update, logging=True):
        if logging:
            self._active_job_log[key].append(update)
            self.meta.update_submissions(self._active_job, self._active_job_log)

    # Allowed on parallel process
    def _warn_job(self, warntext, logging=True):
        self._append_job_log('warnings', warntext, logging=logging)
        print(warntext)

    # Allowed on parallel process
    def _fail_job(self, failtext, logging=True):
        self._update_job_log({
            'finished_time': datetime.datetime.now().isoformat(),
            'status': 'Failed: ' + failtext
        }, logging=logging)
        raise Exception(failtext)



    # =================================================================
    # ======================= Database handling =======================
    # =================================================================

    def surv_exists(self):
        return self.meta._use_files and self.meta.surv_file.exists()

    def read_surv(self, pf=None):
        """Load the surv dataframe containing the particles informations into a file.

        Args:
            pf ([type], optional): File object used for loading. Defaults to None.
        """
        if self.surv_exists():
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.surv_file, 'rb', wait=_db_access_wait_time,
                                     max_lock_time=_db_max_lock_time) as pf:
                        self._surv = pd.read_parquet(pf, engine="pyarrow")
                else:
                    self._surv = pd.read_parquet(pf, engine="pyarrow")
            else:
                raise NotImplementedError
        else:
            return None

    def write_surv(self, pf=None):
        """Save the surv dataframe containing the particles informations into a file.

        Args:
            pf ([type], optional): File object used for saving. Defaults to None.
        """
        if self.meta._use_files and not self.meta._read_only:
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.surv_file, 'wb', wait=_db_access_wait_time) as pf:
                        self._surv.to_parquet(pf, index=True, engine="pyarrow")
                else:
                    pf.truncate(0)  # Delete file contents (to avoid appending)
                    pf.seek(0)      # Move file pointer to start of file
                    self._surv.to_parquet(pf, index=True, engine="pyarrow")
            else:
                raise NotImplementedError

    def da_exists(self):
        return self.meta._use_files and self.meta.da_file.exists()

    def read_da(self, pf=None):
        """Load the da border dataframe into a file.

        Args:
            pf ([type], optional): File object used for Loading. Defaults to None.
        """
        if self.da_exists():
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.da_file, 'rb', wait=_db_access_wait_time,
                                     max_lock_time=_db_max_lock_time) as pf:
                        self._da = pd.read_parquet(pf, engine="pyarrow")
                else:
                    self._da = pd.read_parquet(pf, engine="pyarrow")
            else:
                raise NotImplementedError
        else:
            return None

    def write_da(self, pf=None):
        """Save the da dataframe into a file.

        Args:
            pf ([type], optional): File object used for saving. Defaults to None.
        """
        if self.meta._use_files and not self.meta._read_only:
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.da_file, 'wb', wait=_db_access_wait_time) as pf:
                        self._da.to_parquet(pf, index=True, engine="pyarrow")
                else:
                    pf.truncate(0)  # Delete file contents (to avoid appending)
                    pf.seek(0)      # Move file pointer to start of file
                    self._da.to_parquet(pf, index=True, engine="pyarrow")
            else:
                raise NotImplementedError

    def border_exists(self):
        return self.meta._use_files and self.meta.border_file.exists()

    def read_border(self, pf=None):
        """Loading the da border dataframe into a file.

        Args:
            pf ([type], optional): File object used for loading. Defaults to None.
        """
        if self.border_exists():
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.border_file, 'rb', wait=_db_access_wait_time) as pf:
                        self._border = pd.read_parquet(pf, engine="pyarrow")
                else:
                    self._border = pd.read_parquet(pf, engine="pyarrow")
                for cc in self._border.columns:
                    self._border.loc[self._border[cc]==-1,cc]=np.nan
            else:
                raise NotImplementedError
        else:
            return None

    def write_border(self, pf=None):
        """Save the da border dataframe into a file.

        Args:
            pf ([type], optional): File object used for saving. Defaults to None.
        """
        if self.meta._use_files and not self.meta._read_only:
            if self.meta.db_extension=='parquet':
                self._border.fillna(-1)
                if pf is None:
                    with ProtectFile(self.meta.border_file, 'wb', wait=_db_access_wait_time) as pf:
                        self._border.to_parquet(pf, index=True, engine="pyarrow")
                else:
                    pf.truncate(0)  # Delete file contents (to avoid appending)
                    pf.seek(0)      # Move file pointer to start of file
                    self._border.to_parquet(pf, index=True, engine="pyarrow")
                for cc in self._border.columns:
                    self._border.loc[self._border[cc]==-1,cc]=np.nan
            else:
                raise NotImplementedError

    def da_evol_exists(self):
        return self.meta._use_files and self.meta.da_evol_file.exists()

    def read_da_evol(self, pf=None):
        """Load the da evol file.

        Args:
            pf ([type], optional): File object used for loading. Defaults to None.
        """
        if self.da_evol_exists():
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.da_evol_file, 'rb', wait=_db_access_wait_time,
                                     max_lock_time=_db_max_lock_time) as pf:
                        self._da_evol = pd.read_parquet(pf, engine="pyarrow")
                else:
                    self._da_evol = pd.read_parquet(pf, engine="pyarrow")
            else:
                raise NotImplementedError
        else:
            return None

    def write_da_evol(self, pf=None):
        """Save the da evol file.

        Args:
            pf ([type], optional): File object used for saving. Defaults to None.
        """
        if self.meta._use_files and not self.meta._read_only:
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.da_evol_file, 'wb', wait=_db_access_wait_time) as pf:
                        self._da_evol.to_parquet(pf, index=True, engine="pyarrow")
                else:
                    pf.truncate(0)  # Delete file contents (to avoid appending)
                    pf.seek(0)      # Move file pointer to start of file
                    self._da_evol.to_parquet(pf, index=True, engine="pyarrow")
            else:
                raise NotImplementedError

    def da_model_exists(self):
        return self.meta._use_files and self.meta.da_model_file.exists()

    def read_da_model(self, pf=None):
        """Load the da models parameters into a file.

        Args:
            pf ([type], optional): File object used for loading. Defaults to None.
        """
        if self.da_model_exists():
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.da_model_file, 'rb', wait=_db_access_wait_time,
                                     max_lock_time=_db_max_lock_time) as pf:
                        self._da_model = pd.read_parquet(pf, engine="pyarrow")
                else:
                    self._da_model = pd.read_parquet(pf, engine="pyarrow")
                # Change columns format in order to be saved/loaded
#                 col=[c.split() for c in self._da_model.columns]
#                 for ii in range(len(col)):
#                     col[ii][1]=int(col[ii][1])
#                 self._da_model.columns=pd.MultiIndex.from_tuples(col,names=['model', 'nprm', 'key'])
                for cc in self._da_model.columns:
                    self._da_model.loc[self._da_model[cc]==-1,cc]=np.nan
            else:
                raise NotImplementedError
        else:
            self._da_model = None

    def write_da_model(self, pf=None):
        """Save the da models parameters into a file.

        Args:
            pf ([type], optional): File object used for saving. Defaults to None.
        """
        if self.meta._use_files and not self.meta._read_only:
            if self.meta.db_extension=='parquet':
                self._da_model.fillna(-1)
                # Change columns format in order to be saved/loaded
#                 save_col=self._da_model.columns.copy()
#                 self._da_model.columns=[f'{c[0]} {c[1]} {c[2]}' for c in self._da_model.columns]
                if pf is None:
                    with ProtectFile(self.meta.da_model_file, 'wb', wait=_db_access_wait_time) as pf:
                        self._da_model.to_parquet(pf, index=True, engine="pyarrow")
                else:
                    pf.truncate(0)  # Delete file contents (to avoid appending)
                    pf.seek(0)      # Move file pointer to start of file
                    self._da_model.to_parquet(pf, index=True, engine="pyarrow")
                # Change columns format in order to be saved/loaded
#                 self._da_model.columns=save_col
                for cc in self._da_model.columns:
                    self._da_model.loc[self._da_model[cc]==-1,cc]=np.nan
            else:
                raise NotImplementedError
            
    
    # =================================================================

    def convert_to_radial(self):
        # impossible; only to add dimensions or something like that
        raise NotImplementedError

        
        
        

# Function to cut out islands: only keep the turn number if smaller than the previous one.
# Otherwise, replace by previous turn number.
descend = np.frompyfunc(lambda x, y: y if y < x else x, 2, 1)
    
    
    

# --------------------------------------------------------
def _calculate_radial_evo(data):
    # We select the window in number of turns for which each angle has values
    minturns = np.array([ x[1].min() for x in data.values()]).max()
    maxturns = np.array([ x[1].max() for x in data.values()]).min()
    turns = np.unique([ x for x in np.concatenate([ x[1] for x in data.values()]) if (x >= minturns and x <= maxturns) ])
    angles=np.array(list(data.keys()))
    # initialise numpy array
    result = {turn: np.zeros(len(angles)) for turn in turns }
    for i_ang,ang in enumerate(angles):
        rho_surv = np.flip(data[ang])
        # Shift the value of Nturn slightly up such that, in case of a vertical cliff,
        # the interpolation will assign the largest rho_surv value. This cannot be
        # done for the smallest Nturn as this will give an out-of-bounds error.
        rho_surv[0,1:] += 1e-7
        rho_turns = interpolate.interp1d(rho_surv[0], rho_surv[1], kind='linear')(turns)
        for i_turn, turn in enumerate(turns):
            result[turn][i_ang] = rho_turns[i_turn]
    da = np.zeros(len(turns))
    for i_turn, turn in enumerate(turns):
        da[i_turn] = np.sqrt(
            2/np.pi*integrate.trapezoid(x=angles*np.pi/180, y=result[turn]**2)
        )
    return np.array([turns, da])
    
def _get_raw_da_radial(data):
    angles = np.unique(data['angle'])
    ampcol = 'amplitude'
    da = {}
    for angle in angles:
        datasort = data[data['angle']==angle].sort_values(ampcol)
        # Cut out islands from evolution (i.e. force non-monotonously descending)
        datasort['turns_no_islands'] = descend.accumulate(datasort.turns.astype(object))
        # Keep the values around the places where the number of turns changes
        mask = (datasort.turns_no_islands.diff() < 0 ) | (datasort.turns_no_islands.shift(-1).diff() < 0)
        mask.iloc[-1] = True  # Add the last amplitude step (needed for interpolation later)
        da[angle] = np.array((datasort[mask][ampcol].values,datasort[mask].turns_no_islands.values), dtype=float)
    return da

def get_da_radial(files):
    data = pd.concat([ pd.read_csv(file) for file in files ])
    return np.array([ [ang, vals[0,0]] for ang, vals in _get_raw_da_radial(data).items() ])

def get_da_evo_radial(files):
    data = pd.concat([ pd.read_csv(file) for file in files ])
    return _calculate_radial_evo(_get_raw_da_radial(data))
# --------------------------------------------------------






# --------------------------------------------------------
def concat(df1,df2, **kwargs):
    if df1 is None:
        df1=df2.copy()
#         df1.reset_index(inplace=True)
        for kk,vv in kwargs.items():
            df1[kk]=vv
        return df1
    
    # Check if line exist in df1
    msk=np.array([(df1[kk]==vv) for kk,vv in kwargs.items()]).sum(axis=0) ==len(kwargs)
    # Check if df2  columnn ot in df1
    newcol=[cc for cc in df2.columns if cc not in df1.columns]
    
    df2_srt=df2.dropna(axis=1, how='all')
    
    df1_tbs=df1.loc[~msk,:]
    df1_tbm=df1.loc[ msk,[cc for cc in df1.columns if cc not in kwargs.keys() and cc not in df2_srt.columns]]
    df1_tbm=df1_tbm.dropna(axis=0, how='all')
    
    if df1_tbm.empty:
        # Merge the 2 DataFrame
        for kk,vv in kwargs.items():
            df2_srt[kk]=vv
        return pd.concat([df1_tbs,df2_srt], ignore_index=True)
    else:
        col=np.concatenate([df1.columns,newcol])
        ldf1_tbm=len(df1_tbm)
        ldf2_srt=len(df2_srt)
        # Merge the 2 DataFrame
        dfn_tbm=pd.DataFrame(index=range(max(ldf1_tbm,ldf2_srt)), columns=col)
        dfn_tbm.loc[:ldf1_tbm,df1_tbm.columns]=df1_tbm.values
        dfn_tbm.loc[:ldf2_srt,df2_srt.columns]=df2_srt.values
        
        for kk,vv in kwargs.items():
            dfn_tbm[kk]=vv
        return pd.concat([df1_tbs,dfn_tbm], ignore_index=True)
    
    
def select_rows(df, **kwargs):
    # Check if line exist in df
    msk=np.array([(df[kk]==vv) for kk,vv in kwargs.items()]).sum(axis=0) == len(kwargs)
    return df.loc[msk,:]


def get_border_min(DA,seed,t):
    bdr=select_rows(DA._border,seed=seed,t=t)

    border_min=bdr.loc[:,['ang lower','amp lower','id lower']].copy()
    border_min.rename(columns={'ang lower':'angle', 'amp lower':'amplitude', 'id lower':'id'}, 
                      inplace=True)
    border_min.dropna(inplace=True)
    return border_min


def get_border_max(DA,seed,t):
    bdr=select_rows(DA._border,seed=seed,t=t)

    border_max=bdr.loc[:,['ang upper','amp upper','id upper']].copy()
    border_max.rename(columns={'ang upper':'angle', 'amp upper':'amplitude', 'id upper':'id'}, 
                      inplace=True)
    border_max.dropna(inplace=True)
    return border_max
# --------------------------------------------------------
    
    
    


    
    
    

# Function loading SixDesk/SixDB outputs into XDyna
# --------------------------------------------------------
def load_sixdesk_output(path, study, nemit=None, load_line: bool=False): # TODO: Add reference emitance, if emittance difference from file inform that if BB some results will be wrong
    ## SIXDESK
    ## -----------------------------------
    # Load meta
    meta=pd.read_csv(path+'/'+study+".meta.csv",header=0); meta=meta.set_index('keyname')

    # Load polar
    # tp=pd.read_csv(path+'/'+study+".polar.csv",header=0)
    # polar_seed =tp.loc[:,'seed'].values
    # polar_ang  =tp.loc[:,'angle'].values
    # polar_DA_P =tp.loc[:,'alost2'].values
    # polar_DA_P1=tp.loc[:,'alost1'].values

    # Load surv
    tp=pd.read_csv(path+'/'+study+".surv.csv",header=0)
    surv_seed =tp.loc[:,'seed'].values
    surv_ang  =tp.loc[:,'angle'].values
    surv_amp  =tp.loc[:,'amp'].values
    surv_ntrn1=tp.loc[:,'sturns1'].values
    surv_ntrn2=tp.loc[:,'sturns2'].values


    ## META
    ## -----------------------------------
    if not Path(path,study+'.meta.json').exists():
        # Generate meta class
        sixdb_da = DA(name=study,                                   # Name of the Study
                   path=Path(path),                              # Path to the Study (path/name.meta.json)
                   normalised_emittance=np.float64(meta.loc['emit','value'])*1e-6,  # Normalised emittance: ne or (nex,ney) [m]
                   max_turns=int(meta.loc['turnsl','value']), 
                   nseeds=max(surv_seed),                        # For multiseed study (Default=0)
                   use_files=True)

    else:
        # Load the study metadata"
        sixdb_da = DA(name=study, path=Path(path), use_files=True)


    ## LINE
    ## -----------------------------------
    if sixdb_da.line_file is None and load_line:
        # Define the line
        sixdb_da.madx_file = Path(path,study+".mask")       # MadX File to build the line from
        sixdb_da.line_file = Path(path,study+".line.json")  # Line File path

        if not sixdb_da.line_file.exists():
            # Set label to remove:
            label={
                       "%EMIT_BEAM":np.float64(meta.loc['emit','value']),  # [um]
                       "%NPART":1,
                       "%XING":np.float64(meta.loc['xing','value']),
                   }

            # Unmask the mask
            with open(sixdb_da.madx_file, 'r') as fin:
                data = fin.read()
                for key, value in label.items():
                    print(key, value)
                    data=data.replace(key, str(value))
                with ProtectFile(Path(path,study+".mask.unmasked"), 'w', wait=_db_access_wait_time) as fout:
                    fout.write(data)
            sixdb_da.madx_file = Path(path,study+".mask.unmasked")

            # Build the line from MadX
            sequence= 'lhcb1' if meta.loc['beam','value']=='B1' else 'lhcb2'
            sixdb_da.build_line_from_madx(sequence=sequence,  run_all_seeds= (sixdb_da.meta.nseeds!=0) )
    
    
    ## SURV
    ## -----------------------------------
    if not Path(path,study+'.surv.paquet').exists():
        # Load particle distribution as a polar grid.
        x = surv_amp*np.cos(surv_ang*np.pi/180)
        y = surv_amp*np.sin(surv_ang*np.pi/180)

        sixdb_da._surv = pd.DataFrame(index=range(len(surv_amp)))
        sixdb_da._surv.loc[:,'seed'] = surv_seed
        sixdb_da._surv.loc[:,'ang_xy'] = surv_ang
        sixdb_da._surv.loc[:,'r_xy'] = surv_amp
        sixdb_da._surv.loc[:,'nturns'] = surv_ntrn1
        sixdb_da._surv.loc[:,'x_norm_in'] = x
        sixdb_da._surv.loc[:,'y_norm_in'] = y
        sixdb_da._surv.loc[:,'px_norm_in'] = 0
        sixdb_da._surv.loc[:,'py_norm_in'] = 0
        sixdb_da._surv.loc[:,'zeta_in'] = 0
        sixdb_da._surv.loc[:,'delta_in'] = np.float64(meta.loc['dpini','value'])
        sixdb_da._surv.loc[:,'x_out'] = 0
        sixdb_da._surv.loc[:,'y_out'] = 0
        sixdb_da._surv.loc[:,'px_out'] = 0
        sixdb_da._surv.loc[:,'py_out'] = 0
        sixdb_da._surv.loc[:,'zeta_out'] = 0
        sixdb_da._surv.loc[:,'delta_out'] = 0
        sixdb_da._surv.loc[:,'s_out'] = 0
        sixdb_da._surv.loc[:,'state'] = 1*(surv_ntrn1==sixdb_da.max_turns)
        sixdb_da._surv.loc[:,'submitted'] = True
        sixdb_da._surv.loc[:,'finished'] = True
        sixdb_da.meta.pairs_shift=1
        sixdb_da.meta.pairs_shift_var='x'
        sixdb_da._create_pairs()
        orig = (sixdb_da._surv['paired_to'] == sixdb_da._surv.index)
        sixdb_da._surv.loc[~orig,'nturns'] = surv_ntrn2
        sixdb_da._surv.loc[~orig,'state'] = 1*(surv_ntrn2==sixdb_da.max_turns)
        sixdb_da.write_surv()
        sixdb_da.meta.da_type = 'radial'
        sixdb_da.meta.da_dim = 2
        sixdb_da.meta.r_max = np.max(np.sqrt(x**2 + y**2))
        sixdb_da.meta.ang_min = 0
        sixdb_da.meta.ang_max = 90
        sixdb_da.meta.npart = len(sixdb_da._surv.index)
        
        if nemit is not None:
            warnings.warn(f"A renormalisation of emittances is applyed ([{sixdb_da.nemitt_x},{sixdb_da.nemitt_y}] -> {nemit}). "
                          + "This might lead to error in the analysis especially for Beam-beam simulations.")
            sixdb_da.update_emittance(nemit, update_surv=True)
    
    sixdb_da.meta._store()
    return sixdb_da
# --------------------------------------------------------
