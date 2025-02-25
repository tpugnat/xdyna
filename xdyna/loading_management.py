import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from xaux import ProtectFile

from .da import DA
from .da_meta import _db_access_wait_time


# Function loading SixDesk/SixDB outputs into XDyna
# --------------------------------------------------------
def load_sixdesk_output(path:Path|str, study:str, nemit=None, load_line: bool=False): # TODO: Add reference emitance, if emittance difference from file inform that if BB some results will be wrong
    if isinstance(path, str):
        path = Path(path)
    ## SIXDESK
    ## -----------------------------------
    # Load meta
    meta=pd.read_csv(path / (study+".meta.csv"),header=0); meta=meta.set_index('keyname')

    # Load polar
    # tp=pd.read_csv(path+'/'+study+".polar.csv",header=0)
    # polar_seed =tp.loc[:,'seed'].values
    # polar_ang  =tp.loc[:,'angle'].values
    # polar_DA_P =tp.loc[:,'alost2'].values
    # polar_DA_P1=tp.loc[:,'alost1'].values

    # Load surv
    tp=pd.read_csv(path / (study+".surv.csv"),header=0)
    surv_seed =tp.loc[:,'seed'].values
    surv_ang  =tp.loc[:,'angle'].values
    surv_amp  =tp.loc[:,'amp'].values
    surv_ntrn1=tp.loc[:,'sturns1'].values
    surv_ntrn2=tp.loc[:,'sturns2'].values

    ## META
    ## -----------------------------------
    if not (path / (study+'.meta.json')).exists():
        # Generate meta class
        sixdb_da = DA(name=study,                                   # Name of the Study
                   path=path,                              # Path to the Study (path/name.meta.json)
                   normalised_emittance=np.float64(meta.loc['emit','value'])*1e-6,  # Normalised emittance: ne or (nex,ney) [m]
                   max_turns=int(meta.loc['turnsl','value']), 
                   nseeds=max(surv_seed),                        # For multiseed study (Default=0)
                   use_files=True)
    else:
        # Load the study metadata"
        sixdb_da = DA(name=study, path=path, use_files=True)

    ## LINE
    ## -----------------------------------
    if sixdb_da.line_file is None and load_line:
        # Define the line
        sixdb_da.madx_file = path / (study+".mask")       # MadX File to build the line from
        sixdb_da.line_file = path / (study+".line.json")  # Line File path

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
                with ProtectFile(path / (study+".mask.unmasked"), 'w', wait=_db_access_wait_time) as fout:
                    fout.write(data)
            sixdb_da.madx_file = path / (study+".mask.unmasked")

            # Build the line from MadX
            sequence= 'lhcb1' if meta.loc['beam','value']=='B1' else 'lhcb2'
            sixdb_da.build_line_from_madx(sequence=sequence,  run_all_seeds= (sixdb_da.meta.nseeds!=0) )
    
    
    ## SURV
    ## -----------------------------------
    if not (path / (study+'.surv.paquet')).exists():
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


# Function loading XDyna outputs
# --------------------------------------------------------
def load_data(study,path):
    if not Path(path, study+".meta.json").exists():
        print('Generate Xdyna output from SixDesk')
        return load_sixdesk_output(str(path),study)
    else:
        print('Load Xdyna output from Xdyna')
        return DA(name=study,path=path, use_files=True)
# --------------------------------------------------------
