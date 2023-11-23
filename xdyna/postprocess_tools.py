import numpy as np
import pandas as pd
from scipy import interpolate, integrate
from scipy.special import lambertw as W



# Open border interpolation
# --------------------------------------------------------
def trapz(x, y, xrange):
    """
    Return the integral using the trapezoidal rule for open border.
    Works for not constant step too.
    """
    x=np.array(x); y=np.array(y); sort=np.argsort(x); 
    x=x[sort]; y=y[sort]
#     D=integrate.trapezoid(x=x*np.pi/180, y=np.ones(x.size))
#     return np.sqrt( 2/np.pi*integrate.trapezoid(x=x*np.pi/180, y=y**2) )
#     return np.sqrt( integrate.trapezoid(x=x*np.pi/180, y=y**2)/D )
#     return integrate.trapezoid(x=x*np.pi/180, y=y)/D
    
    res =y[0]*(x[0]-xrange[0]) + y[-1]*(xrange[1]-x[-1])          # Lower and upper open border schema
    res+= (0.5)*( ( y[1:] + y[:-1] )*(x[1:] - x[:-1]) ).sum()     # Close border schema
    return res

    
def simpson(x, y, xrange):
    """
    Return the integral using the simpson's 1/3 rule for open border.
    Works for not constant step too.
    """
    if len(y)>=3 and (len(y) % 2)==1:
        x=np.array(x); y=np.array(y); sort=np.argsort(x); 
        x=x[sort]; y=y[sort]

        res =(23*y[ 0]-16*y[ 1]+5*y[ 2])*(x[ 0]-xrange[0])/12          # Lower open border schema
        res+=(23*y[-1]-16*y[-2]+5*y[-3])*(xrange[1]-x[-1])/12          # Upper open border schema
        
        # Constant stepsize
#         res+= ( (y[0:-1:2]+4*y[1::2]+y[2::2])*(x[2::2] - x[0:-1:2]) ).sum()/6     # Close border schema
        
        # Different stepsize
        h1 =(x[1::2] - x[0:-1:2])
        h2 =(x[2::2] - x[1::2])
        res+=( (y[0:-1:2]*h2*(2*h1**2-h2*(h2-h1)) + y[1::2]*(h1+h2)**3 + y[2::2]*h1*(2*h2**2+h1*(h2-h1)))/(6*h1*h2) ).sum()

        return res
    else:
        return 0

    
def alter_simpson(x, y, xrange): # used to be called compute_da
    """
    Return the integral using the alternative simpson rule for open border.
    Does not works for not constant step.
    """
    if len(y)>6:
        x=np.array(x); y=np.array(y); sort=np.argsort(x); 
        x=x[sort]; y=y[sort]

        res =(23*y[ 0]-16*y[ 1]+5*y[ 2])*(x[ 0]-xrange[0])/12          # Lower open border schema
        res+=(23*y[-1]-16*y[-2]+5*y[-3])*(xrange[1]-x[-1])/12          # Upper open border schema
        wght=np.ones(len(y)); wght[0]=wght[-1]=3/8 ; wght[1]=wght[-2]=-16/12 ; wght[2]=wght[-3]=5/12 ; 
        res+= ( y*wght ).sum()*(x[1]-x[0])                             # Close border schema
        return res
    else:
        return 0

    
# Just call trapz with y=amplitude**2 instead and sqrt the result
# def trapz_norm2(x, y, xrange):
#     """
#     Return the quadratic mean using the trapezoidal rule for open border.
#     """
#     x=np.array(x); y=np.array(y); sort=np.argsort(x); 
#     x=x[sort]; y=y[sort]**2
    
#     # Trapz open border
#     res =y[0]*(x[0]-xrange[0]) + y[-1]*(xrange[1]-x[-1])
#     res+= (0.5)*( ( y[1:] + y[:-1] )*(x[1:] - x[:-1]) ).sum()
#     return np.sqrt( res/(xrange[1]-xrange[0]) )
# --------------------------------------------------------
    
    
    

# Compute average DA
# --------------------------------------------------------
def compute_da_1D(x, y, xrange, interp=trapz): # used to be called compute_da
    """
    Return the arithmetic average. Default interpolator: trapz.
    """
    return interp(x, y, xrange)/(xrange[1]-xrange[0])

    
def compute_da_2D(x, y, xrange, interp=trapz):
    """
    Return the quadratic average. Default interpolator: trapz.
    """
    return np.sqrt( interp(x, y**2, xrange)/(xrange[1]-xrange[0]) )

    
def compute_da_4D(x, y, xrange, interp=trapz):
    """
    Return the 4D average. Default interpolator: trapz.
    """
    return interp(x, (y**4)*np.sin(np.pi*x/90), xrange)**(1/4)
# --------------------------------------------------------
    
    
    

# Extrapolation from set of points
# --------------------------------------------------------
def fit_DA(x, y, xrange):
    """ 1D fit function f(angle) with 'angle' in [deg].
    """
    xmin=min([xrange[0],min(x)]); xmax=max([xrange[1],max(x)])
    
    x=np.array(x); y=np.array(y); sort=np.argsort(x)
    x=x[sort]; y=y[sort]; 
    
    # Increase the range of the fitting in order to prevent errors
    if xmin<-135 and xmax>135:
        ymid=((y[0]-y[-1])*(180-x[-1]))/((x[0]+360-x[-1]))+y[-1]
        x=np.append([-180],x); y=np.append([ymid],y)
        x=np.append(x,[ 180]); y=np.append(y,[ymid])
    else:
        x=np.append([np.floor(xmin)-5],x); y=np.append([y[0]],y)
        x=np.append(x,[np.ceil(xmax)+5]);  y=np.append(y,[y[-1]])
    return interpolate.interp1d(x, y)
# --------------------------------------------------------
    
    
    

# DA vs Turns Models
# --------------------------------------------------------
# Taken from https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.22.104003
def Model_2(N, rho=1, K=1, N0=1):            # Eq. 20
    return rho * ( K/( 2*np.exp(1)*np.log(N/N0) ) )**K 
Model_2_default  ={'rho':1, 'K':1, 'N0':1}
Model_2_boundary ={'rho':[1e-10,np.inf], 'K':[0.01,2], 'N0':[1,np.inf]}
def Model_2_mask(N, rho=1, K=1, N0=1):
    return [True for nn in N]

def Model_2b(N, btilde=1, K=1, N0=1, B=1):   # Eq. 35a
    return btilde / ( B*np.log(N/N0) )**K      
Model_2b_default ={'btilde':1, 'K':1, 'N0':1} #, 'B':1}
Model_2b_boundary={'btilde':[1e-10,np.inf], 'K':[0.01,2], 'N0':[1,np.inf]} #, 'B':[1e-10,1e5]}
def Model_2b_mask(N, btilde=1, K=1, N0=1, B=1):
    return [True for nn in N]

def Model_2n(N, b=1, K=1, N0=1):             # Eq. 2 from Frederik
    return b / ( np.log(N/N0) )**K      
Model_2n_default ={'b':1, 'K':1, 'N0':1}
Model_2n_boundary={'b':[1e-10,np.inf], 'K':[0.01,2], 'N0':[1,np.inf]}
def Model_2n_mask(N, b=1, K=1, N0=1):
    return [True for nn in N]



def Model_4(N, rho=1, K=1, lmbd=0.5):        # Eq. 23
    return rho / ( -(2*np.exp(1)*lmbd) * np.real(W( (-1/(2*np.exp(1)*lmbd)) * (rho/6)**(1/K) * (8*N/7)**(-1/(lmbd*K)) ,k=-1)) )**K  
Model_4_default  ={'rho':1, 'K':1} #, 'lmbd':0.5
Model_4_boundary ={'rho':[1e-10,1e10], 'K':[0.01,2]} #, 'lmbd':[1e-10,1e10]}
def Model_4_mask(N, rho=1, K=1, lmbd=0.5):
    return np.imag(W( (-1/(2*np.exp(1)*lmbd)) * (rho/6)**(1/K) * (8*N/7)**(-1/(lmbd*K)) ,k=-1))==0

def Model_4b(N, btilde=1, K=1, N0=1, B=1):   # Eq. 35c
    return btilde / (-(0.5*K*B) * np.real(W( (-2/(K*B)) * (N/N0)**(-2/K) ,k=-1)) )**K  
Model_4b_default ={'btilde':1, 'K':1, 'N0':1} #, 'B':1}
Model_4b_boundary={'btilde':[1e-10,np.inf], 'K':[0.01,2], 'N0':[1,np.inf]} #, 'B':[1e-10,1e10]}
def Model_4b_mask(N, btilde=1, K=1, N0=1, B=1):
    return np.imag(W( (-2/(K*B)) * (N/N0)**(-2/K) ,k=-1))==0

def Model_4n(N, rho=1, K=1, mu=1):           # Eq. 4 from Frederik
    return rho / (- np.real(W( (mu*N)**(-2/K) ,k=-1)) )**K  
Model_4n_default ={'rho':1, 'K':1, 'mu':1}
Model_4n_boundary={'rho':[1e-10,np.inf], 'K':[0.01,2], 'mu':[1e-10,1e10]}
def Model_4n_mask(N, rho=1, K=1, mu=1):
    return np.imag(W( (mu*N)**(-2/K) ,k=-1))==0


def Model_user_mask(N, **kwarg):
    return [True for nn in N]


def select_model(model,model_default={},model_boundary={},model_mask=Model_user_mask,name='user'):
    if isinstance(model,str):
#         print(f"{type(model)=}")
#         print(f"{type(model_default)=}")
#         print(f"{type(model_boundary)=}")
#         print(f"{type(model_mask)=}")
        model=model.lower(); mdefault =model_default.copy(); 
        mboundary=model_boundary.copy(); mmask=model_mask;
        if ('model_2' ==model) or ('2' ==model):
            name='2';  
            model=Model_2;   
            mdefault =Model_2_default.copy();   
            mboundary=Model_2_boundary.copy(); 
            mmask    =Model_2_mask;
#                 keys=[k for k in Model_2_default.keys()];
        if ('model_2b'==model) or ('2b'==model):
            name='2b'; 
            model=Model_2b;  
            mdefault =Model_2b_default.copy();  
            mboundary=Model_2b_boundary.copy();
            mmask    =Model_2b_mask;
#                 keys=[k for k in Model_2b_default.keys()];
        if ('model_2n'==model) or ('2n'==model):
            name='2n'; 
            model=Model_2n;  
            mdefault =Model_2n_default.copy();  
            mboundary=Model_2n_boundary.copy();
            mmask    =Model_2n_mask;
#                 keys=[k for k in Model_2n_default.keys()];
        if ('model_4' ==model) or ('4' ==model):
            name='4';  
            model=Model_4;   
            mdefault =Model_4_default.copy();   
            mboundary=Model_4_boundary.copy();
            mmask    =Model_4_mask;
#                 keys=[k for k in Model_4_default.keys()];
        elif ('model_4b'==model) or ('4b'==model):
            name='4b'; 
            model=Model_4b;  
            mdefault =Model_4b_default.copy();  
            mboundary=Model_4b_boundary.copy();
            mmask    =Model_4b_mask;
#                 keys=[k for k in Model_4b_default.keys()];
        elif ('model_4n'==model) or ('4n'==model):
            name='4n'; 
            model=Model_4n;  
            mdefault =Model_4n_default.copy();  
            mboundary=Model_4n_boundary.copy();
            mmask    =Model_4n_mask;
#                 keys=[k for k in Model_4n_default.keys()];
    elif not isinstance(model_default,dict) or not isinstance(model_boundary,dict):
        raise ValueError('If you give your own model, give model_default and model_boundary parameter as dictionaries.')
    return name, model, mdefault, mboundary, mmask
            
            
    model=model.lower()
    if isinstance(model,str):
        if model in ['model_2','model_2b','model_2n','model_4','model_4b','model_4n',
                     '2','2b','2n','4','4b','4n']:
            if ('model_2' ==model) or ('2' ==model):
                name='2';  model=Model_2;   keys=[k for k in Model_2_default.keys()];
            if ('model_2b'==model) or ('2b'==model):
                name='2b'; model=Model_2b;  
            if ('model_2n'==model) or ('2n'==model):
                name='2n'; model=Model_2n;  
            if ('model_4' ==model) or ('4' ==model):
                name='4';  model=Model_4;   
            if ('model_4b'==model) or ('4b'==model):
                name='4b'; model=Model_4b;  
            if ('model_4n'==model) or ('4n'==model):
                name='4n'; model=Model_4n;  
    elif keys is None:
            raise ValueError('Please specify the parameters name as keys.')
# --------------------------------------------------------

    
    
    

# DA raw estimation
# --------------------------------------------------------
def _da_raw(data,at_turn):
    # Detect range to look at the DA border
    losses =data.nturns<at_turn
    loss=data.loc[ losses,:]; min_loss=min(loss.amplitude)
    surv=data.loc[~losses,:]; max_surv=max(surv.amplitude)
    min_amplitude = min([min_loss,max_surv])-2
    max_amplitude = max([min_loss,max_surv])+2

    # Get a raw DA estimation from losses
    list_angles=np.unique(data['round_angle'])
#     border_max={'id':[],'angle':[],'amplitude':[]}
#     border_min={'id':[],'angle':[],'amplitude':[]}
    border_max=pd.DataFrame({},index=range(len(list_angles)),columns=['id','angle','amplitude'])
    border_min=pd.DataFrame({},index=range(len(list_angles)),columns=['id','angle','amplitude'])
    for idx,ang in enumerate(list_angles):
        # Select angulare slice
        section=data.loc[data.round_angle==ang,:]

        # Identify losses and surviving particles
        losses =section.nturns<at_turn
        section_loss=section.loc[ losses,:]; section_loss=section_loss.loc[section_loss.amplitude<=max_amplitude,:]
        section_surv=section.loc[~losses,:]; section_surv=section_surv.loc[section_surv.amplitude>=min_amplitude,:]

        # Detect DA boundary
        if not section_loss.empty and not section_surv.empty:
            min_amplitude_loss=min(section_loss.amplitude)
#             border_max['amplitude'].append(min_amplitude_loss)
#             border_max['angle'].append(section_loss.loc[section_loss.amplitude==min_amplitude_loss,'angle'].values[0])
#             border_max['id'].append(section_loss.loc[section_loss.amplitude==min_amplitude_loss,'id'].values[0])
            min_section_loss=section_loss.loc[section_loss.amplitude==min_amplitude_loss,:].reset_index()
            border_max.loc[idx,['amplitude','angle','id']]=[min_section_loss.amplitude[0],min_section_loss.angle[0],
                                                            min_section_loss.id[0]]

            mask = section_surv.amplitude<min_amplitude_loss
            if any(mask):
                max_amplitude_surv=max(section_surv.amplitude[mask])
#                 border_min['amplitude'].append(max_amplitude_surv)
#                 border_min['angle'].append(section_surv.loc[section_surv.amplitude==max_amplitude_surv,'angle'].values[0])
#                 border_min['id'].append(section_surv.loc[section_surv.amplitude==max_amplitude_surv,'id'].values[0])
                max_section_surv=section_surv.loc[section_surv.amplitude==max_amplitude_surv,:].reset_index()
                border_min.loc[idx,['amplitude','angle','id']]=[max_section_surv.amplitude[0],max_section_surv.angle[0],
                                                                max_section_surv.id[0]]

        elif not section_loss.empty:
            min_amplitude_loss=min(section_loss.amplitude)
#             border_max['amplitude'].append(min_amplitude_loss)
#             border_max['angle'].append(section_loss.loc[section_loss.amplitude==min_amplitude_loss,'angle'].values[0])
#             border_max['id'].append(section_loss.loc[section_loss.amplitude==min_amplitude_loss,'id'].values[0])
            min_section_loss=section_loss.loc[section_loss.amplitude==min_amplitude_loss,:].reset_index()
            border_max.loc[idx,['amplitude','angle','id']]=[min_section_loss.amplitude[0],min_section_loss.angle[0],
                                                            min_section_loss.id[0]]

        elif not section_surv.empty:
            max_amplitude_surv=max(section_surv.amplitude)
#             border_min['amplitude'].append(max_amplitude_surv)
#             border_min['angle'].append(section_surv.loc[section_surv.amplitude==max_amplitude_surv,'angle'].values[0])
#             border_min['id'].append(section_surv.loc[section_surv.amplitude==max_amplitude_surv,'id'].values[0])
            max_section_surv=section_surv.loc[section_surv.amplitude==max_amplitude_surv,:].reset_index()
            border_min.loc[idx,['amplitude','angle','id']]=[max_section_surv.amplitude[0],max_section_surv.angle[0],
                                                            max_section_surv.id[0]]
#     return pd.DataFrame(border_min), pd.DataFrame(border_max)
    return border_min.dropna().reset_index(), border_max.dropna().reset_index()
# --------------------------------------------------------

    
    
    

# DA smoothing procedure
# --------------------------------------------------------
# Not allowed on parallel process
def _da_smoothing(data,raw_border_min,raw_border_max,at_turn,removed=pd.DataFrame(columns=['id']),
                  DA_lim_min=None,DA_lim_max=None, active_warmup=True, ang_range=None):
    
    data['id']= data.index
    if 'angle' not in data.columns or 'amplitude' not in data.columns:
        data['angle']      = np.arctan2(data['y'],data['x'])*180/np.pi
        data['amplitude']  = np.sqrt(data['x']**2+data['y']**2)
    if ang_range is None:
        ang_range=(min(data.angle),max(data.angle))

    # Check if raw border cross each other
    raw_fit_min=fit_DA(raw_border_min.angle, raw_border_min.amplitude, ang_range)
    raw_fit_max=fit_DA(raw_border_max.angle, raw_border_max.amplitude, ang_range)
    out_min=raw_border_min.loc[raw_border_min.amplitude>=raw_fit_max(raw_border_min.angle)]
    out_max=raw_border_max.loc[raw_border_max.amplitude<=raw_fit_min(raw_border_max.angle)]
    if not out_min.empty or not out_max.empty:
        raise ValueError(f'Both border are crossing each other at t={int(at_turn):d}:\n'+
                         f'  * Losses in min DA border:\n{out_max}\n\n  * Min DA border outside max DA border:\n{out_min}')

    # Apply upper and lower limit to the data
    if DA_lim_min is not None:
        data=data.loc[data.amplitude>=DA_lim_min,:]
    if DA_lim_max is not None:
        data=data.loc[data.amplitude<=DA_lim_max,:]
        
    if removed.empty:
        surv=data.loc[data.nturns>=at_turn,:]
    else:
        surv=data.loc[data.nturns>=at_turn,:].drop(index=removed.loc[removed.nturns>=at_turn,'id']);
    loss=data.loc[data.nturns< at_turn,:];
    tmp_border_min=raw_border_min
    tmp_border_max=raw_border_max

    # Add extra particle from warming up the DA borders
    if active_warmup:
        tmp_fit_min=fit_DA(tmp_border_min.angle, tmp_border_min.amplitude, ang_range)
        tmp_fit_max=fit_DA(tmp_border_max.angle, tmp_border_max.amplitude, ang_range)

        # Check surv particles bellow and within a distance of 2 sigma from min border
        dist_to_border_min=tmp_fit_min(surv.angle)-surv.amplitude
        surv_in_DA=surv.loc[(dist_to_border_min>=0) & (dist_to_border_min<2),:]
#         if DA_lim_min is None:
#             surv_in_DA=surv.loc[(dist_to_border_min>=0) & (dist_to_border_min<2),:]
#         else:
#             surv_in_DA=surv.loc[(dist_to_border_min>=0) & (dist_to_border_min<2) & (surv.amplitude>=DA_lim_min),:]

        # Check surv particles higher and within a distance of 2 sigma from max border
        dist_to_border_max=loss.amplitude-tmp_fit_max(loss.angle)
        loss_ex_DA=loss.loc[(dist_to_border_max<2),:]

        # Add particle to the DA border
        tmp_border_min=surv_in_DA.loc[:,['id','angle','amplitude']]
        tmp_border_max=loss_ex_DA.loc[:,['id','angle','amplitude']]

        # Remove angle duplicate
        angs,nmb=np.unique(tmp_border_min.angle,return_counts=True)
        for a in angs[nmb>1]:
            mask=tmp_border_min.angle==a
            max_amp=max(tmp_border_min.loc[mask,'amplitude'])
            id_remove=tmp_border_min.loc[mask & (tmp_border_min.amplitude<max_amp),'id']
            tmp_border_min.drop(index=id_remove,inplace=True)
        angs,nmb=np.unique(tmp_border_min.angle,return_counts=True)

        angs,nmb=np.unique(tmp_border_max.angle,return_counts=True)
        for a in angs[nmb>1]:
            mask=tmp_border_max.angle==a
            min_amp=min(tmp_border_max.loc[mask,'amplitude'])
            id_remove=tmp_border_max.loc[mask & (tmp_border_max.amplitude>min_amp),'id']
            tmp_border_max.drop(index=id_remove,inplace=True)
        angs,nmb=np.unique(tmp_border_max.angle,return_counts=True)

    # Smoothing procedure
    it=0
    continue_smoothing=True
    while continue_smoothing:
        continue_smoothing=False

        tmp_fit_min=fit_DA(tmp_border_min.angle, tmp_border_min.amplitude, ang_range)
        tmp_fit_max=fit_DA(tmp_border_max.angle, tmp_border_max.amplitude, ang_range)
        tmp_da_min =compute_da_1D(tmp_border_min.angle, tmp_border_min.amplitude,ang_range)
        tmp_da_max =compute_da_1D(tmp_border_max.angle, tmp_border_max.amplitude,ang_range)

#             print('''
#             # Check if surviving particles outside min DA border can be added to the border 
#             # without having losses inside.
#             ''')
        cand=surv.loc[surv.amplitude>tmp_fit_min(surv.angle),:]
        for idx, c in cand.iterrows():
            new_border_min=pd.concat([ tmp_border_min,cand.loc[[idx],['id','angle','amplitude']] ])
#                 new_border_min=pd.DataFrame({'id':np.append(tmp_border_min.id,[c.id]),
#                                              'angle':np.append(tmp_border_min.angle,[c.angle]),
#                                              'amplitude':np.append(tmp_border_min.amplitude,[c.amplitude])})
#                 new_border_min.set_index('id',drop=False,inplace=True)

            # Remove angle duplicate
            angs,nmb=np.unique(new_border_min.angle,return_counts=True)
            for a in angs[nmb>1]:
                mask=new_border_min.angle==a
                max_amp=max(new_border_min.loc[mask,'amplitude'])
                id_remove=new_border_min.loc[mask & (new_border_min.amplitude<max_amp),'id']
                new_border_min.drop(index=id_remove,inplace=True)

            angs,nmb=np.unique(new_border_min.angle,return_counts=True)

            new_fit_min=fit_DA(new_border_min.angle, new_border_min.amplitude, ang_range)

            loss_in_DA = loss.loc[loss.amplitude<=new_fit_min(loss.angle),:]
            if loss_in_DA.empty:
                # If candidate lower than max DA boundary, it is atomaticaly added
                if c.amplitude<tmp_fit_max(c.angle):
                    tmp_border_min=new_border_min
                    tmp_fit_min=fit_DA(tmp_border_min.angle, tmp_border_min.amplitude, ang_range)
                    continue_smoothing=True
#                         print(f'Normal add:\n{c}')
                # Else we must check if adding at least one point to max DA boundary
                # allow it to not cross anymore max DA boundary
                else:
                    loss_strict=loss.loc[loss.amplitude>c.amplitude].copy()
#                         loss_strict=loss_strict.loc[loss_strict.amplitude>c.amplitude]

                    loss_strict['dist']=np.abs( loss_strict.amplitude*np.exp(1j*np.pi*loss_strict.angle/180)
                                         -c.amplitude*np.exp(1j*np.pi*c.angle/180))
                    loss_index=np.array(loss_strict.index[np.argsort(loss_strict['dist'])])

                    iloss=0
                    while iloss<min([5,len(loss_index)]) and c.amplitude>tmp_fit_max(c.angle):
                        idx=loss_index[iloss]
                        new_border_max=pd.concat([ tmp_border_max, loss_strict.loc[[idx],['id','angle','amplitude']] ])
#                             l=loss_strict.loc[idx,:]
#                             new_border_max=pd.DataFrame({'id':np.append(tmp_border_max.id,[l.id]),
#                                                          'angle':np.append(tmp_border_max.angle,[l.angle]),
#                                                          'amplitude':np.append(tmp_border_max.amplitude,[l.amplitude])})
#                             new_border_max.set_index('id',drop=False,inplace=True)

                        new_fit_max=fit_DA(new_border_max.angle, new_border_max.amplitude, ang_range)

                        new_border_max=loss.loc[loss.amplitude<=new_fit_max(loss.angle),['id','angle','amplitude']]
                        new_fit_max=fit_DA(new_border_max.angle, new_border_max.amplitude, ang_range)
                        if c.amplitude<new_fit_max(c.angle):
                            tmp_border_min=new_border_min
                            tmp_border_max=new_border_max
                            tmp_fit_min=fit_DA(tmp_border_min.angle, tmp_border_min.amplitude, ang_range)
                            tmp_fit_max=fit_DA(tmp_border_max.angle, tmp_border_max.amplitude, ang_range)
                            continue_smoothing=True
#                                 print(f'Specific add:\n{c}')

                        iloss+=1


#             print('''
        # Check if some min DA border particles could be removed without having losses 
        # inside.
#             ''')
        surv_in_da=surv.loc[surv.amplitude<=tmp_fit_min(surv.angle),:]
        index=tmp_border_min.index[np.argsort(tmp_border_min.amplitude)]
        for idx in index:
            new_border_min=tmp_border_min.drop(index=idx)
            new_fit_min=fit_DA(new_border_min.angle, new_border_min.amplitude, ang_range)
            new_da_min =compute_da_1D(new_border_min.angle, new_border_min.amplitude,ang_range)

            surv_ex_DA = surv_in_da.loc[surv_in_da.amplitude>new_fit_min(surv_in_da.angle),:]
            loss_in_DA = loss.loc[loss.amplitude<=new_fit_min(loss.angle),:]

            if loss_in_DA.empty and surv_ex_DA.empty and new_da_min>tmp_da_min and len(new_border_min)>3:
#                     print(f'\nRemove:\n{tmp_border_min.loc[idx,:]}\n')
                tmp_border_min=new_border_min
                tmp_fit_min=fit_DA(tmp_border_min.angle, tmp_border_min.amplitude, ang_range)
                continue_smoothing=True
#             tmp_border_min.reset_index(inplace=True, drop=True)


#             print('''
        # Check if some max DA border particles could be removed without cross min DA border
#             ''')
        surv_in_da=surv.loc[surv.amplitude<=tmp_fit_min(surv.angle),:]
        index=tmp_border_max.index[np.flip(np.argsort(tmp_border_max.amplitude))]
        for idx in index:
            new_border_max=tmp_border_max.drop(index=idx)
            new_fit_max=fit_DA(new_border_max.angle, new_border_max.amplitude, ang_range)
            new_da_max =compute_da_1D(new_border_max.angle, new_border_max.amplitude,ang_range)

            surv_ex_DA = surv_in_da.loc[surv_in_da.amplitude>=new_fit_max(surv_in_da.angle),:]
            loss_in_DA = loss.loc[loss.amplitude<new_fit_max(loss.angle),:]

            if loss_in_DA.empty and surv_ex_DA.empty and new_da_max<tmp_da_max and len(new_border_max)>3:
#                     print(f'\nRemove:\n{tmp_border_max.loc[idx,:]}\n')
                tmp_border_max=new_border_max
                tmp_fit_max=fit_DA(tmp_border_max.angle, tmp_border_max.amplitude, ang_range)
                tmp_da_max =compute_da_1D(tmp_border_max.angle, tmp_border_max.amplitude,ang_range)
                continue_smoothing=True
#             tmp_border_max.reset_index(inplace=True, drop=True)

#             print(it)
        it+=1
    
    return tmp_border_min,tmp_border_max
# --------------------------------------------------------