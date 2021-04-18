import math
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from multiprocessing import Pool,Process,Queue
#from make_insar_downsample import rms_block_demean

def make_insar_downsample_nonrecursive(xinsar,yinsar,zinsar,Nmin,Nres_min,Nres_max,method):

    r1 = 10
    if method == 'mean':
        [xout, yout, zout, Npt, rms_out, xx1, xx2, yy1, yy2] = quad_decomp_mean_nonrecursive(xinsar, yinsar, zinsar, r1,
                                                                                 Nres_min, Nres_max)
    Ndata = len(zout)
    Nint = 0
    while (Ndata < Nmin):
        N1 = len(zout)
        r1 = r1 * 0.85
        if method == 'mean':
            [xout, yout, zout, Npt, rms_out, xx1, xx2, yy1, yy2] = quad_decomp_mean_nonrecursive(xinsar, yinsar, zinsar, r1,
                                                                                    Nres_min, Nres_max)
        Ndata = len(zout)
        N2 = len(zout)
        Nint = Nint + 1
        if ((N2 > 0.8 * Nmin) & ((N2-N1) < 0.005 * N1)):
            break
    xout = np.array(xout)
    yout = np.array(yout)
    zout = np.array(zout)
    Npt = np.array(Npt)
    rms_out = np.array(rms_out)
    xx1 = np.array(xx1)
    xx2 = np.array(xx2)
    yy1 = np.array(yy1)
    yy2 = np.array(yy2)

    return xout, yout, zout, Npt, rms_out, xx1, xx2, yy1, yy2


def quad_decomp_mean_nonrecursive_v1(xin,yin,zin,threshold,Nres_min,Nres_max):


    xx1=[]; xx2=[]; yy1=[]; yy2=[]; Ndata=[]; rms_out=[]
    xout=[]; yout=[]; zout=[];
    
    # Initialize for the first layer
    num_node_this_layer=1;
    nx=[]; nx.append(int(np.size(xin)))
    ny=[]; ny.append(int(np.size(yin)))

    # grid boundary index
    idx1=[]; idx1.append(int(0))
    idx2=[]; idx2.append(int(nx[0]-1))
    idy1=[]; idy1.append(int(0))
    idy2=[]; idy2.append(int(ny[0]-1))

    max_depth=int(np.ceil(np.log(nx[0]*ny[0]/(Nres_min*Nres_min))/np.log(4)))+1
    print('max depth: ',max_depth)

    rms_default = 10
    r_good_default = 0.2
    
    for i in range(0,max_depth):
        # i: this layer
        # do quad-tree and get the number of children nodes
        # For all parents node in this depth level (will have at most 4^i)
        nx_next=[]; ny_next=[];
        idx1_next=[]; idx2_next=[]; 
        idy1_next=[]; idy2_next=[];
        
        print(i,num_node_this_layer)
        num_node_next_layer=0
        for j in range(0,num_node_this_layer):
            
            [rms1, N1, r1_good, x1_out, y1_out, z1_out] = rms_block_demean(xin[idx1[j]:idx2[j]+1], yin[idy1[j]:idy2[j]+1], zin[idy1[j]:idy2[j]+1,idx1[j]:idx2[j]+1], Nres_min, Nres_max)
            if (((nx[j] <= Nres_min) | (ny[j]<=Nres_min) | (rms1 <= threshold)) & (N1 > 0) & (r1_good > r_good_default)):
                
                xout.append(x1_out)
                yout.append(y1_out)
                zout.append(z1_out)
                Ndata.append(N1)
                rms_out.append(rms1)
                xx1.append(xin[idx1[j]])
                xx2.append(xin[idx2[j]])
                yy1.append(yin[idy1[j]])
                yy2.append(yin[idy2[j]])

            #elif ((N1 > 0) & (r1_good > r_good_default)):
            else:
                # If this node is not leaf node, will have 4 silblings
                # Index of grid edge
                tmp_x=idx1[j]+int(np.floor(nx[j]/2))
                tmp_y=idy1[j]+int(np.floor(ny[j]/2))
                idx1_next.extend([idx1[j], tmp_x, idx1[j], tmp_x]); 
                idx2_next.extend([tmp_x, idx2[j], tmp_x, idx2[j]]); 
                idy1_next.extend([idy1[j], idy1[j], tmp_y, tmp_y]); 
                idy2_next.extend([tmp_y, tmp_y, idy2[j], idy2[j]]);
                # print(idx1[j],tmp_x,idx2[j])
                # Counts
                nx_next.extend([np.size(xin[idx1[j]:tmp_x+1]), np.size(xin[tmp_x:idx2[j]+1]), np.size(xin[idx1[j]:tmp_x+1]), np.size(xin[tmp_x:idx2[j]+1])])
                ny_next.extend([np.size(yin[idy1[j]:tmp_y+1]), np.size(yin[idy1[j]:tmp_y+1]), np.size(yin[tmp_y:idy2[j]+1]), np.size(yin[tmp_y:idy2[j]+1])])
                num_node_next_layer = num_node_next_layer + 4
                 

        # reset every layer
        num_node_this_layer = num_node_next_layer
        nx = nx_next
        ny = ny_next
        # Index of grid edge
        idx1 = idx1_next
        idx2 = idx2_next
        idy1 = idy1_next
        idy2 = idy2_next

    return xout,yout,zout,Ndata,rms_out,xx1,xx2,yy1,yy2


def calculation_kernel(j,indx_layer,max_depth,xin,yin,zin,Nx,Ny,Nres_min,Nres_max,threshold,r_good_default):
    # Which depth am I in ? 
    for d in range(0, max_depth+1):
        if j < indx_layer[d]:
            # depth of this layer
            d0 = d-1
            break

    #print(d0)
    # Get local index: index in this layer, instead of cumulative index
    j0 = j - int(indx_layer[d0])
    #print('Global, Local index :', j, j0)

    # Based on i, get the grid boundary of this grid 
    # Each edge is divided into 2^d0 pieces
    # Define the dx, and dy location of this piece
    dx = int(j0 % math.pow(2,d0))
    dy = int(j0 / math.pow(2,d0))
    #print('dx, dy :', dx, dy)
    
    idx1 = int(dx*Nx/math.pow(2,d0))
    idx2 = int((dx+1)*Nx/math.pow(2,d0))
    idy1 = int(dy*Ny/math.pow(2,d0))
    idy2 = int((dy+1)*Ny/math.pow(2,d0))
    #print(idx1[j],idx2[j],idy1[j],idy2[j])

    xx1 = xin[idx1]
    xx2 = xin[idx2]
    yy1 = yin[idy1]
    yy2 = yin[idy2]
    
    nx = np.size(xin[idx1:idx2+1])
    ny = np.size(yin[idy1:idy2+1])
   
    [rms, N, r_good, xout, yout, zout] = rms_block_demean(xin[idx1:idx2+1], yin[idy1:idy2+1], 
                                        zin[idy1:idy2+1,idx1:idx2+1], Nres_min, Nres_max)
   
    if (((nx <= Nres_min) | (ny<=Nres_min) | (rms <= threshold)) & (N > 0) & (r_good > r_good_default)):
        is_leaf = 1
    else:
        is_leaf = 0

    # q.put((j,rms, N, r_good, xout, yout, zout, idx1, idx2, idy1, idy2, xx1, xx2, yy1, yy2, is_leaf))
    # return a tuple
    return [rms, N, r_good, xout, yout, zout, idx1, idx2, idy1, idy2, xx1, xx2, yy1, yy2, is_leaf]

# Another version prep for GPU opepration 
# Take out all unsupported
class DownSample_InSAR(Dataset):
  """ Dataset for cutting templates
  """
  def __init__(self,indx_layer,max_depth,xin,yin,zin,Nx,Ny,Nres_min,Nres_max,threshold,r_good_default):
    self.indx_layer = indx_layer
    self.max_depth = max_depth
    self.xin = xin
    self.yin = yin
    self.zin = zin
    self.Nx = Nx
    self.Ny = Ny
    self.Nres_min = Nres_min
    self.Nres_max = Nres_max
    self.threshold =threshold
    self.r_good_default = r_good_default

  def __getitem__(self, index):
    return calculation_kernel(index,self.indx_layer,self.max_depth,self.xin,self.yin,self.zin,self.Nx,self.Ny,self.Nres_min,self.Nres_max,self.threshold,self.r_good_default)
  
  def __len__(self):
    return len(self.xin) # just to return a value, no meaning



def quad_decomp_mean_nonrecursive(xin,yin,zin,threshold,Nres_min,Nres_max):

    rms_default = 10
    r_good_default = 0.2
    
    # Allocate array 
    # Nx, Ny index, not number
    Nx = np.size(xin)-1
    Ny = np.size(yin)-1
    max_depth = int(np.ceil(np.log(int(np.size(xin))*int(np.size(yin))/(Nres_min*Nres_min))/np.log(4)))+1
    indx_layer = np.zeros(max_depth+1)
    indx_layer[0]=0
    for i in range(0, max_depth):
        indx_layer[i+1] = indx_layer[i] + math.pow(4,i)

    max_size = int(indx_layer[max_depth])    
    print(Nx, Ny, max_size)

    is_leaf = np.zeros(max_size)

    idx1 = np.zeros(max_size,dtype=int)
    idx2 = np.zeros(max_size,dtype=int)
    idy1 = np.zeros(max_size,dtype=int)
    idy2 = np.zeros(max_size,dtype=int)
    
    xout = np.zeros(max_size)
    yout = np.zeros(max_size)
    zout = np.zeros(max_size)
    N = np.zeros(max_size,dtype=int)
    r_good = np.zeros(max_size)
    rms = np.zeros(max_size)
    
    xx1 = np.zeros(max_size)
    xx2 = np.zeros(max_size)
    yy1 = np.zeros(max_size)
    yy2 = np.zeros(max_size)
    
    # print(indx_layer)
    # In this case, no concepts of layer
    # Calculate values in all grid size
    #for j in range(0, max_size):
    #   [rms[j], N[j], r_good[j], xout[j], yout[j], zout[j], idx1[j], idx2[j], idy1[j], idy2[j], xx1[j], xx2[j], yy1[j], yy2[j], is_leaf[j]]=calculation_kernel(j,indx_layer,max_depth,xin,yin,zin,Nx,Ny,Nres_min,Nres_max,threshold,r_good_default)

    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    dataset = DownSample_InSAR(indx_layer,max_depth,xin,yin,zin,Nx,Ny,Nres_min,Nres_max,threshold,r_good_default)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=None)
    for j,[rmsi,Ni,r_goodi,xouti,youti,zouti,idx1i,idx2i,idy1i,idy2i,xx1i,xx2i,yy1i,yy2i,is_leafi] in enumerate(dataloader):
        [rms[j], N[j], r_good[j], xout[j], yout[j], zout[j], idx1[j], idx2[j], idy1[j], idy2[j], xx1[j], xx2[j], yy1[j], yy2[j], is_leaf[j]] = [rmsi,Ni,r_goodi,xouti,youti,zouti,idx1i,idx2i,idy1i,idy2i,xx1i,xx2i,yy1i,yy2i,is_leafi]  


    #pool = Pool(12)
    #pool.map_async(save_syn_batch,range(1000))
    #pool.close()
    #pool.join()

    # Reset leaf-node label based on parent node
    for j in range(0, max_size):
        if is_leaf[j]==1:
            for d in range(0, max_depth+1):
                if j < indx_layer[d]:
                    # depth of this layer
                    d0 = d-1
                    break

            #j0 = j - int(indx_layer[d0])
            #dx = int(j0 % math.pow(2,d0))
            #dy = int(j0 / math.pow(2,d0))

            # need to optimize this
            for jj in range(0,j):
                # if found leaf node, break
                isparent = (idx1[j]>=idx1[jj]) & (idx2[j]<=idx2[jj]) & (idy1[j]>=idy1[jj]) & (idy2[j]<=idy2[jj])
                if isparent & (is_leaf[jj]==1):
                    is_leaf[j]=0
                    break

    xout = xout[is_leaf==1]
    yout = yout[is_leaf==1]
    zout = zout[is_leaf==1]
    N = N[is_leaf==1]
    rms = rms[is_leaf==1]
    xx1 = xx1[is_leaf==1]
    xx2 = xx2[is_leaf==1]
    yy1 = yy1[is_leaf==1]
    yy2 = yy2[is_leaf==1]

    return xout,yout,zout,N,rms,xx1,xx2,yy1,yy2


# Another version of rms_block_demean that use math library
# Prep for GPU use
def rms_block_demean(x,y,z,Nres_min,Nres_max):
    [xx, yy] = np.meshgrid(x,y)
    indx_good = ~np.isnan(z)
    [nx, ny] = np.shape(z)
    n_block = nx*ny
    xdata = xx[indx_good]
    ydata = yy[indx_good]
    zdata = z[indx_good]
    Ngood = np.shape(zdata)[0]
    r_good = float(Ngood)/n_block
    if (Ngood > 0):
        xout = np.mean(xdata)
        yout = np.mean(ydata)
        zout = np.mean(zdata)
        lx = np.shape(np.unique(x))[0]
        ly = np.shape(np.unique(y))[0]
        if ((Ngood<=3) | (lx<=Nres_min) | (ly<=Nres_min)):
            rms_out = 0
        elif ((Ngood > 5) & ((lx>2) & (lx < Nres_max)) & ((ly>2) & (ly < Nres_max))):
            zz = zdata
            zzfit = np.mean(zz)
            dz = zz-zzfit
            rms_out = np.sqrt(np.sum(dz**2)/Ngood)
        else:
            rms_out = 1000
    else:
        xout = np.nan
        yout = np.nan
        zout = np.nan
        rms_out = 0
    return rms_out, Ngood, r_good, xout, yout, zout

