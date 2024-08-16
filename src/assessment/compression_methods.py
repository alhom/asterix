import numpy as np
import vdf_extract
import mlp_compress
import shutil
import struct
import sys,os
import warnings
import numpy as np
import ctypes
import pyzfp,zlib
import mlp_compress
import tools
import pytools

VDF_TYPE_SIZE=4

def sparsify(vdf,sparsity):
    # vdf[vdf<sparsity]=0.0
    return
    
# MLP with fourier features
def reconstruct_cid_fourier_mlp(f, cid,sparsity):
    order = 48
    epochs = 50
    hidden_layers=[75,50,50,10]
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    reconstructed_vdf,bytes_used =mlp_compress.compress_mlp_from_vec(
                 vdf.flatten(), order, epochs,np.array(hidden_layers,dtype=np.uint64) , nx, sparsity) 
    reconstructed_vdf = np.reshape(reconstructed_vdf,(nx, ny, nz))
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    original_vdf_size=vdf.size*VDF_TYPE_SIZE
    final_vdf_size=bytes_used
    cm_ratio=original_vdf_size/final_vdf_size
    return cid, np.array(final_vdf, dtype=np.float32),cm_ratio

# MLP
def reconstruct_cid_mlp(f, cid,sparsity):
    order = 0
    epochs = 50
    hidden_layers=[75,50,50,10]
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    reconstructed_vdf,bytes_used =mlp_compress.compress_mlp_from_vec(
                 vdf.flatten(), order, epochs,np.array(hidden_layers,dtype=np.uint64) , nx, sparsity) 
    reconstructed_vdf = np.reshape(reconstructed_vdf,(nx, ny, nz))
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    original_vdf_size=vdf.size*VDF_TYPE_SIZE
    final_vdf_size=bytes_used
    cm_ratio=original_vdf_size/final_vdf_size
    return cid, np.array(final_vdf, dtype=np.float32),cm_ratio

# ZFP
def reconstruct_cid_zfp(f, cid,sparsity):
    tolerance = 1e-13
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    compressed_vdf = pyzfp.compress(vdf, tolerance=tolerance)
    reconstructed_vdf = pyzfp.decompress(compressed_vdf,vdf.shape,vdf.dtype,tolerance)
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    original_vdf_size=vdf.size*VDF_TYPE_SIZE
    final_vdf_size=compressed_vdf.size*VDF_TYPE_SIZE
    cm_ratio=original_vdf_size/final_vdf_size
    return cid, np.array(final_vdf, dtype=np.float32), cm_ratio

# Spherical Harmonics
def reconstruct_cid_sph(f, cid,sparsity):
    degree =10 
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    reconstructed_vdf=mlp_compress.compress_sph_from_vec(vdf.flatten(),degree,nx)
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,np.shape(vdf),order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32),1


# Octree
def reconstruct_cid_oct(f, cid,sparsity):
    from juliacall import Main as jl
    max_indexes, vdf ,len= vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    jl.Pkg.activate("src/jl_env")
    jl.Pkg.instantiate()
    jl.include("src/octree.jl")

    residual, padimg, tree, compress_ratio = jl.VDFOctreeApprox.compress_conservative(vdf_3d, maxiter=1000, tol=0.025, verbose=False, errtype="ltwo")

    reco = (padimg-residual).to_numpy()

    reconstructed_vdf=np.array(reco,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,np.shape(vdf),order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf

    original_vdf_size=vdf.size*VDF_TYPE_SIZE
    compressed_vdf_size=(3*length(tree) + VDF_TYPE_SIZE*length(tree)*length(tree[1].data.c))

    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32), original_vdf_size/compressed_vdf_size


# PCA
def reconstruct_cid_pca(f, cid,sparsity):
    from sklearn.decomposition import PCA
    n =10 
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    vdf[vdf<sparsity]=sparsity
    vdf = np.log10(vdf)
    arr=vdf.copy()
    arr = arr.reshape(arr.shape[0], -1)
    cov_matrix = np.cov(arr, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    pca = PCA(n_components=n)
    compressed = pca.fit_transform(arr)
    #reconstruct the data
    recon = pca.inverse_transform(compressed)
    nx,ny,nz=np.shape(vdf)
    recon=np.reshape(recon,(nx,ny,nz))
    reconstructed_vdf = 10 ** recon
    reconstructed_vdf[reconstructed_vdf <= sparsity] = 0
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,np.shape(vdf),order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32),1


#CNN
def reconstruct_cid_cnn(f, cid,sparsity):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(16)
            self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm3d(32)
            self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm3d(64)
            self.conv4 = nn.Conv3d(64, 1, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.conv4(x)
            return x

    def train_and_reconstruct(input_array, num_epochs=30, learning_rate=0.001, batch_size=32):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Move input tensor to device
        model = CNN().to(device) 
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
        for epoch in range(num_epochs):
            for i in range(0, input_tensor.size(0), batch_size):
                optimizer.zero_grad()
                batch_input = input_tensor[i:i+batch_size]
                output_tensor = model(batch_input)
                loss = criterion(output_tensor, batch_input)
                loss.backward()
                optimizer.step()
    
        with torch.no_grad():
            output_tensor = model(input_tensor)
        reconstructed_array = output_tensor.squeeze(0).squeeze(0).cpu().numpy()
    
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size = (param_size + buffer_size)   
        return reconstructed_array, size
    
    epochs=10
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    vdf[vdf<sparsity]=sparsity
    vdf = np.log10(vdf)
    input_array=vdf
    reconstructed_vdf,total_size= train_and_reconstruct(input_array,epochs)
    reconstructed_vdf = 10 ** reconstructed_vdf
    reconstructed_vdf[reconstructed_vdf <= sparsity] = 0
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,np.shape(vdf),order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32),1


# GMM
def reconstruct_cid_gmm(f, cid,sparsity):
    n_pop=5
    norm_range=300
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    means,weights,covs,norm_unit=tools.run_gmm(vdf,n_pop,norm_range)
    n_bins=nx
    v_min,v_max=0,nx
    reconstructed_vdf=tools.reconstruct_vdf(n_pop,means,covs,weights,n_bins,v_min,v_max)
    reconstructed_vdf=reconstructed_vdf*norm_unit*norm_range
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,np.shape(vdf),order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32),1

#DWT
def reconstruct_cid_dwt_new(f, cid, sparsity):
    import pywt

    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    threshold = sparsity

    def dwt_quantize_coeffs(c, quantization_dtype = None):

        if quantization_dtype == None:
            return c, [0, 1]
        
        if quantization_dtype.kind == 'u':
            # print(c)
            # cmax, coffset = np.nanmax(c[np.isfinite(c)]), np.nanmin(c[np.isfinite(c)])
            cmax, coffset = np.nanmax(c), np.nanmin(c)
            # print(cmax, coffset)

            crange = cmax-coffset
            dtype_max = np.iinfo(quantization_dtype).max - 1
            dtype_min = np.iinfo(quantization_dtype).min
            mask = ~np.isfinite(c)
            c = (c - coffset)/(crange) * dtype_max
            c[mask] = np.iinfo(quantization_dtype).max
            c = c.astype(quantization_dtype)
            
            return c, (coffset, crange)
        elif quantization_dtype.kind == 'f':
            cmax, coffset = np.nanmax(c), np.nanmin(c)
            # print(cmax, coffset)

            crange = cmax-coffset
            dtype_max = np.finfo(quantization_dtype).max
            dtype_min = 0

            c = (c - coffset)/(crange) * dtype_max
            c[~np.isfinite(c)] = 0 #np.finfo(quantization_dtype).max

            return c.astype(quantization_dtype), (coffset, crange)

    def dwt_dequantize_coeffs(c, quantization_dtype = None, qdat_tuple = (0,1)):
        if quantization_dtype == None:
            return c
        c = c.astype(np.float32)
        if quantization_dtype.kind == 'u':
            mask = c == np.iinfo(quantization_dtype).max
            c = c/(np.iinfo(quantization_dtype).max-1)
            c = c*qdat_tuple[1] + qdat_tuple[0]
            c[mask] = qdat_tuple[0]
        elif quantization_dtype.kind == 'f':
            c = c/(np.finfo(quantization_dtype).max)
            c = c*qdat_tuple[1] + qdat_tuple[0]
        return c

    def dwt_threshold_coeffs(c, threshold = 1e-16, high = 1e-15):

        def soft_threshold(x, threshold = 1e-16):
            return np.sign(x) * np.maximum(np.abs(x)-threshold, 0)

        def hard_threshold(x, threshold = 1e-16):
            x[x < threshold] = 0
            return x

        c = soft_threshold(c, threshold=threshold)
        # print('value_low', threshold, 'value_high', high)
        # c = pywt.threshold_firm(c, value_low=threshold, value_high=high)
        # c = hard_threshold(c, threshold=threshold)

        return c

    #run-length encoding
    rle_type = np.dtype(np.uint32)
    def rle(c):
        # print(c.shape, c.dtype)
        rle_list_of_tuples = []
        runcount = 0
        v0 = 0
        init = True
        for v in np.nditer(c):
            if v == v0:
                runcount += 1
            else:
                if not init:
                    if runcount > np.iinfo(rle_type).max:
                        print("Error: runcount overflow")
                    rle_list_of_tuples.append((runcount, v))
                    v0 = v
                    runcount = 0

            init = False

        return rle_list_of_tuples

    # print(pywt.wavelist(kind='discrete'))

    vdf_3d = vdf.copy().astype(np.float64)

    loga = False
    offset = np.float64(1)
    if loga:
        vdf_3d[vdf_3d<0]=0
        # print(np.nanmax(vdf_3d))
        vdf_3d = np.log10(vdf_3d + offset)
        # print(np.nanmax(vdf_3d))
    if loga:
        norm = np.nanmax(vdf_3d)
    else:
        norm = np.nanmax(vdf_3d)
    print("norm", norm)
    # norm = 1
    vdf_3d /= norm
    orig_shape = vdf_3d.shape
    if loga:
        pass
        # vdf_3d[np.isinf(vdf_3d)] = -np.inf
        # vdf_3d[np.isnan(vdf_3d)] = 0

    comp_type = np.dtype(np.uint16)
    # comp_type = None
    # quant = np.iinfo(comp_type).max/3




    # print(np.nanmax(vdf_3d),np.nanmin(vdf_3d[np.isfinite(vdf_3d)]), norm)

    threshold = np.float64(1e-16)
    wavelet = 'db9'#'bior1.3'
    # wavelet = 'bior1.3'
    # wavelet = 'db9'

    dwtn_mlevel = pywt.dwtn_max_level(vdf_3d.shape,wavelet)
    level_delta = 0
    discard_level = dwtn_mlevel+1
    print("Decomposing to ", max(dwtn_mlevel-level_delta,0), "levels out of ", dwtn_mlevel, 'discarding at level', discard_level )
    coeffs3 = pywt.wavedecn(vdf_3d,wavelet=wavelet, level = max(dwtn_mlevel-level_delta,0))

    # print(coeffs3)
    coeffs3_comp = coeffs3.copy()

    print(type(coeffs3_comp))

    zeros = 0
    nonzeros = 0


    dtype0 = np.dtype(np.float32)
    dtype = np.dtype(np.float32)
    if loga:
        threshold = np.log10(threshold+offset)/norm
    else:
        threshold = threshold/norm
    print(dtype0.itemsize)

    coeffs3_comp_qdat = {}

    for i,a in enumerate(coeffs3_comp):
        
        if(type(a) == type(np.ndarray(1))):
            # print(type(a), a.dtype)
            # print(a)
            
            # print(coeffs3_comp[i])
            if loga:
                coeffs3_comp[i] = dwt_threshold_coeffs(coeffs3_comp[i], 1e-17,1e-17)
            else:
                coeffs3_comp[i] = dwt_threshold_coeffs(coeffs3_comp[i], threshold=threshold, high=threshold*1000)
            # coeffs3_comp[i] = coeffs3_comp[i].astype(dtype)
            coeffs3_comp[i], coeffs3_comp_qdat[i] = dwt_quantize_coeffs(coeffs3_comp[i], comp_type)
            # print(coeffs3_comp[i])
            mask = np.abs(coeffs3_comp[i]) == 0#< threshold
            zeros += np.sum(mask)
            nonzeros += np.sum(~mask)
            # nonzeros += np.prod(a.shape)
            if not loga:
                coeffs3_comp[i][mask] = 0
        else:
            coeffs3_comp_qdat.setdefault(i,{})
            print('dict at level',i)
            for k,v in a.items():
                if i >= discard_level:
                    print('discarding at level', i)
                    coeffs3_comp[i][k] = np.zeros_like(coeffs3_comp[i][k])
                else:
                    if loga:
                        coeffs3_comp[i][k] = dwt_threshold_coeffs(coeffs3_comp[i][k], threshold = 1e-17, high = 1e-17)
                    else:
                        coeffs3_comp[i][k] = dwt_threshold_coeffs(coeffs3_comp[i][k], threshold=threshold, high=threshold*1000)

                # coeffs3_comp[i][k] = coeffs3_comp[i][k].astype(dtype)
                coeffs3_comp[i][k], coeffs3_comp_qdat[i][k] = dwt_quantize_coeffs(coeffs3_comp[i][k], quantization_dtype=comp_type)

                mask = np.abs(coeffs3_comp[i][k]) == 0# < threshold
                
                if not loga:
                    coeffs3_comp[i][k][mask] = 0
                zeros += np.sum(mask)
                nonzeros += np.sum(~mask)

    print("number of zeros:", zeros, "nonzeros:", nonzeros)


    rle_bytes = 0
    rle0 =rle(coeffs3_comp[0])
    # print()
    # print("len rle0:",len(rle0),"manual calculation of RLE bytes for first:", len(rle0)*(rle_type.itemsize+comp_type.itemsize),"sys bytes:", sys.getsizeof(rle0))

    for i,a in enumerate(coeffs3_comp):
        if(type(a) == type(np.ndarray(1))):
            rle0 = rle(coeffs3_comp[i])
            morebytes = len(rle0)*(rle_type.itemsize+comp_type.itemsize)
            # print('rle bytes added: ', morebytes, 'for i',i)
            rle_bytes += morebytes
            coeffs3_comp[i] = dwt_dequantize_coeffs(coeffs3_comp[i], quantization_dtype=comp_type, qdat_tuple=coeffs3_comp_qdat[i])
        else:
            for k,v in a.items():
                rle0 = rle(coeffs3_comp[i][k])
                morebytes = len(rle0)*(rle_type.itemsize+comp_type.itemsize)
                # print('rle bytes added: ', morebytes, 'for i,k', i,k)
                rle_bytes += morebytes
                coeffs3_comp[i][k] = dwt_dequantize_coeffs(coeffs3_comp[i][k], quantization_dtype=comp_type, qdat_tuple=coeffs3_comp_qdat[i][k])

    vdf_rec = pywt.waverecn(coeffs3_comp,wavelet=wavelet)*norm
    # print(np.min(vdf_rec),np.max(vdf_rec))
    if loga:
        vdf_rec = 10**vdf_rec.astype(np.float64) - offset

    # print(np.min(vdf_rec),np.max(vdf_rec))
    vdf_bytes_inflated = np.prod(vdf_3d.shape)*4
    print('Bytes in RLE-encoded coefficients:', rle_bytes, 'bytes in vdf (inflated):',  vdf_bytes_inflated, 'bytes in sparse VDF', true_mem)
    # print('vdf_rec',vdf_rec)
    print('Nonzero bytes in coefficients', nonzeros*comp_type.itemsize)

    # compression = np.prod(vdf_3d.shape)/nonzeros
    # if comp_type is not None:
    #     compression *= dtype0.itemsize/comp_type.itemsize

    compression = rle_bytes/vdf_bytes_inflated

    # volume_compressed = 0
    # for k,v in coeffs3_compress.items():
    #     volume_compressed += sys.getsizeof(v)
    # volume_orig = 0
    # for k,v in coeffs3.items():
    #     volume_orig += sys.getsizeof(v)
    # compression = volume_orig/volume_compressed

    print("compression (naive):", compression)
    print("compression (true):", rle_bytes/true_mem)

    # print(np.min(vdf_rec),np.max(vdf_rec))
    # if ~loga:
    project_tools.plot_vdfs(vdf,vdf_rec.astype(dtype0))
    project_tools.print_comparison_stats(vdf,vdf_rec.astype(dtype0))


# DWT-old
def reconstruct_cid_dwt(f, cid,sparsity):
    import pywt
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    threshold = sparsity
    orig_shape = vdf.shape
    vdf[np.isnan(vdf)] = 0
    comp_type = np.float32
    norm = 1# np.nanmax(vdf)/quant
    vdf /= norm
    vdf[vdf<0]=0
    wavelet = 'db4'#'bior1.3'
    dwtn_mlevel = pywt.dwtn_max_level(vdf.shape,wavelet)
    level_delta = 2
    coeffs3 = pywt.wavedecn(vdf,wavelet=wavelet, level = dwtn_mlevel-2)
    coeffs3_comp = coeffs3.copy()
    zeros = 0
    nonzeros = 0
    for i,a in enumerate(coeffs3_comp):
        print(type(a))
        zero_app = False
        # print(a.shape)
        if(type(a) == type(np.ndarray(1))):
            coeffs3_comp[i] = a
            mask = np.abs(a) < threshold
            zeros += np.sum(mask)
            nonzeros += np.sum(~mask)
            # nonzeros += np.prod(a.shape)
            coeffs3_comp[i][mask] = 0
        else:
            for k,v in a.items():
                mask = np.abs(v) < threshold
                coeffs3_comp[i][k] = v
                coeffs3_comp[i][k][mask] = 0
                zeros += np.sum(mask)
                nonzeros += np.sum(~mask)

    reconstructed_vdf = pywt.waverecn(coeffs3_comp,wavelet=wavelet)*norm    
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,np.shape(vdf),order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32),1


# DCT
def reconstruct_cid_dct(f, cid,sparsity):
    from scipy.fft import dctn, idctn
    blocksize = 8
    keep_n = 4
    max_indexes, vdf,len = vdf_extract.extract(f, cid,sparsity)
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz
    orig_shape = vdf.shape
    vdf[np.isnan(vdf)] = 0

    paddings = (np.ceil(np.array(vdf.shape)/8)).astype(int)*8 - vdf.shape
    paddings = ((0,paddings[0]),(0,paddings[1]),(0,paddings[2]))
    vdf = np.pad(vdf, paddings)

    block_data = np.zeros_like(vdf)
    for i in range(0,vdf.shape[0], blocksize):
        for j in range(0, vdf.shape[1], blocksize):
            for k in range(0, vdf.shape[2], blocksize):
                block_data[i:i+blocksize,j:j+blocksize, k:k+blocksize] = dctn(vdf[i:i+blocksize,j:j+blocksize, k:k+blocksize])

    zeroed = np.zeros_like(block_data)
    for i in range(keep_n):
        for j in range(keep_n):
            for k in range(keep_n):
                zeroed[i::blocksize,j::blocksize,k::blocksize] = block_data[i::blocksize,j::blocksize,k::blocksize]


    volume_compressed = np.prod(keep_n*np.ceil(np.array(vdf.shape)/8))
    volume_orig = np.prod(vdf.shape)
    compression = volume_orig/volume_compressed

    vdf_rec = np.zeros_like(vdf)
    for i in range(0,vdf.shape[0], blocksize):
        for j in range(0, vdf.shape[1], blocksize):
            for k in range(0, vdf.shape[2], blocksize):
                vdf_rec[i:i+blocksize,j:j+blocksize, k:k+blocksize] = idctn(zeroed[i:i+blocksize,j:j+blocksize, k:k+blocksize])

    reconstructed_vdf = vdf_rec[0:orig_shape[0],0:orig_shape[1],0:orig_shape[2]]
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    reconstructed_vdf= np.reshape(reconstructed_vdf,orig_shape,order='C')
    mesh = f.get_velocity_mesh_size()
    final_vdf = np.zeros((int(4 * mesh[0]), int(4 * mesh[1]), int(4 * mesh[2])))
    final_vdf[
        max_indexes[0] - len : max_indexes[0] + len,
        max_indexes[1] - len : max_indexes[1] + len,
        max_indexes[2] - len : max_indexes[2] + len,
    ] = reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32),1

def reconstruct_cid_vqvae(f, cid,sparsity):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import vqvae.vqvae_tools as vq
    model_checkpoint='/home/kstppd/dev/asterix/model_state_70.ptch'
    size = f.get_velocity_mesh_size()
    WID = f.get_WID()
    vmesh_size=size*WID
    vmesh_size=np.array(vmesh_size,dtype=int)
    
    tolerance = 1e-13
    bulk_v_loc, vdf,len = vdf_extract.extract(f, cid,sparsity)
    vdf=vdf_extract.pad_array(vdf,vmesh_size,0.0)
    vdf[vdf<sparsity]=sparsity
    vdf = np.log10(vdf)
    vdf_max_val=vdf.max();
    vdf_min_val=vdf.min();
    vdf = (vdf - vdf.min()) / (vdf.max() - vdf.min())
    nx, ny, nz = np.shape(vdf)
    assert nx == ny == nz

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_ema = True
    model_args = {
        "in_channels": 1,
        "num_hiddens": 128,
        "num_downsampling_layers": 2,
        "num_residual_layers": 2,
        "num_residual_hiddens": 32,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "use_ema": use_ema,
        "decay": 0.99,
        "epsilon": 1e-5,
    }
    model = vq.VQVAE(**model_args).to(device)
    ckpt=torch.load(model_checkpoint)
    new_ckpt = {}
    for k, v in ckpt.items():
        new_ckpt[k.replace('module.', '', 1)] = v
    model.load_state_dict(new_ckpt)
    model.eval() 
    with torch.no_grad():
        vdf=np.array(vdf,dtype=np.float32)
        input = torch.from_numpy(vdf).unsqueeze(0).unsqueeze(0).to(device)
        out = model(input)
        reconstructed_vdf = out["x_recon"].cpu().numpy().squeeze()
        
    origin=np.array(vmesh_size)//2
    offset=bulk_v_loc-origin
    reconstructed_vdf=np.roll(reconstructed_vdf,offset,(0,1,2))
    reconstructed_vdf=reconstructed_vdf*(vdf_max_val - vdf_min_val) + vdf_min_val
    reconstructed_vdf = 10 ** reconstructed_vdf
    reconstructed_vdf=np.array(reconstructed_vdf,dtype=np.double)
    final_vdf=reconstructed_vdf
    sparsify(final_vdf,sparsity)
    return cid, np.array(final_vdf, dtype=np.float32),1


