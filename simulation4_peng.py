
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.sparse import csgraph
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Circle


MPH_TO_MPS = 0.44704
lp0 = 0.005
RADIUS = 20*lp0
steplen = 3.86e-4

def nan_gaussian_blur(Z, sigma_pix=0.8):
    """NaN-aware Gaussian blur (for prettier fields)."""
    from scipy.signal import convolve2d
    if sigma_pix <= 0 or Z is None:
        return Z
    wrad = int(np.ceil(3*sigma_pix))
    y, x = np.mgrid[-wrad:wrad+1, -wrad:wrad+1]
    G = np.exp(-(x**2+y**2)/(2*sigma_pix**2)); G /= G.sum()
    M = np.isfinite(Z).astype(float)
    Z0 = np.where(np.isfinite(Z), Z, 0.0)
    num = convolve2d(Z0, G, mode='same', boundary='symm')
    den = convolve2d(M,  G, mode='same', boundary='symm')
    out = num / np.maximum(den, np.finfo(float).eps)
    out[den==0] = np.nan
    return out

def nearest_node(coords, xy):
    return int(np.argmin(np.linalg.norm(coords - np.array(xy)[None,:], axis=1)))

def build_adjacency(edges_from, edges_to, N):
    from collections import defaultdict
    nbrs = defaultdict(list)
    for u,v in zip(edges_from, edges_to):
        nbrs[u].append(v); nbrs[v].append(u)
    return nbrs

# -------------------------- geometry tests for obstacles --------------------------

def seg_intersects_rect(p0, p1, cx, cy, w, h):
    xmin, xmax = cx - w/2, cx + w/2
    ymin, ymax = cy - h/2, cy + h/2
    x0,y0 = p0; x1,y1 = p1
    if max(x0,x1)<xmin or min(x0,x1)>xmax or max(y0,y1)<ymin or min(y0,y1)>ymax:
        return False
    dx, dy = x1-x0, y1-y0
    if dx==0 and dy==0:
        return (xmin<=x0<=xmax) and (ymin<=y0<=ymax)
    # project rect center to segment and test inside rect
    t = ((cx-x0)*dx + (cy-y0)*dy)/(dx*dx+dy*dy)
    t = np.clip(t, 0.0, 1.0)
    xc, yc = x0 + t*dx, y0 + t*dy
    return (xmin<=xc<=xmax) and (ymin<=yc<=ymax)

def seg_intersects_circle(p0, p1, center, radius):
    """True if the segment p0-p1 intersects or lies fully within a circle."""
    c = np.asarray(center, float); a = np.asarray(p0, float); b = np.asarray(p1, float)
    ab = b - a
    denom = np.dot(ab, ab)
    if denom <= 0:
        return np.linalg.norm(a - c) <= radius
    t = np.clip(np.dot(c - a, ab) / denom, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(closest - c) <= radius

def line_intersects_segment(p0, p1, a, b):
    def cross(u,v): return u[0]*v[1]-u[1]*v[0]
    p0 = np.asarray(p0); p1=np.asarray(p1); a=np.asarray(a); b=np.asarray(b)
    d1 = p1-p0; d2 = b-a
    D = cross(d1,d2)
    if np.isclose(D,0.0): return False
    t = cross(a-p0, d2)/D
    u = cross(a-p0, d1)/D
    return (0<=t<=1) and (0<=u<=1)

# -------------------------- lattices --------------------------

def build_tri_lattice(L=2.0, W=np.sqrt(3.0), lp0=lp0):
    """hexagonal lattice with spacing 2*lp0"""
    dx = 2*lp0
    dy = np.sqrt(3.0)*lp0
    nx = int(round(L/dx)+1)
    ny = int(round(W/dy)+1)
    N  = nx*ny
    coords = np.zeros((N,2), float)
    def idx(i,j): return j*nx+i
    for j in range(ny):
        for i in range(nx):
            n = idx(i,j)
            coords[n] = (i*dx + (lp0 if (j%2)==1 else 0.0), j*dy)
    edges = []
    for j in range(ny):
        for i in range(nx):
            n = idx(i,j)
            if i<nx-1: edges.append((n, idx(i+1,j)))
            if j<ny-1:
                edges.append((n, idx(i,j+1)))
                if (j%2)==1 and i<nx-1:
                    edges.append((n, idx(i+1,j+1)))
                elif (j%2)==0 and i>0:
                    edges.append((n, idx(i-1,j+1)))
    edges = np.array(edges, int)
    elen = 2*lp0
    return coords, edges, lp0, elen

def build_square_lattice(L=2.0, W=2.0, lp0=lp0):
    """square lattice with spacing 2*lp0."""
    dx = 2*lp0; dy=dx
    nx = int(round(L/dx)+1)
    ny = int(round(W/dy)+1)
    N  = nx*ny
    coords = np.zeros((N,2), float)
    def idx(i,j): return j*nx+i
    for j in range(ny):
        for i in range(nx):
            coords[idx(i,j)] = (i*dx, j*dy)
    edges=[]
    for j in range(ny):
        for i in range(nx):
            n=idx(i,j)
            if i<nx-1: edges.append((n, idx(i+1,j)))
            if j<ny-1: edges.append((n, idx(i,j+1)))
    edges=np.array(edges,int)
    elen = 2*lp0
    return coords, edges, lp0, elen

# -------------------------- KCL assembly --------------------------

import numpy as np
from scipy import sparse

def edge_orientation(p0, p1):
    """
    Return one of {'horiz','vert','diag_br','diag_bl'} based on the closest
    of the four canonical axes.
    """
    v = p1 - p0
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 'horiz'
    vhat = v / n

    b_h = np.array([1.0, 0.0])                 # horizontal
    b_v = np.array([0.0, 1.0])                 # vertical
    b_br = np.array([0.5, -np.sqrt(3)/2.0])    # "\" TL→BR (tri lattice)
    b_bl = np.array([-0.5,  np.sqrt(3)/2.0])   # "/"  TR→BL

    bases = [b_h, b_v, b_br, b_bl]
    dots = [abs(np.dot(vhat, b)) for b in bases]
    return ['horiz', 'vert', 'diag_br', 'diag_bl'][int(np.argmax(dots))]


def apply_dirichlet_row_replace(M, nodes, value):
    """Overwrite rows for Dirichlet nodes to phi=value."""
    ML = M.tolil(copy=True); b = np.zeros(M.shape[0], float)
    for n in nodes:
        ML.rows[n] = [n]
        ML.data[n] = [1.0]
        b[n] = value
    return ML.tocsr(), b

def solve_with_component_pinning(M, b, pinned_nodes):
    """
    Solve robustly by pinning one node per disconnected component
    that lacks a Dirichlet pin. Removes nullspaces cleanly.
    """
    from scipy.sparse import csgraph
    from scipy.sparse.linalg import spsolve

    A = M.copy().tocsr()
    A.setdiag(0); A.eliminate_zeros()
    A = (A != 0).astype(int)
    n_comp, labels = csgraph.connected_components(A, directed=False)

    pinned_labels = set(labels[np.atleast_1d(pinned_nodes)])
    ML = M.tolil(copy=True)
    for comp in range(n_comp):
        if comp in pinned_labels:
            continue
        n = int(np.where(labels == comp)[0][0])
        ML.rows[n] = [n]; ML.data[n] = [1.0]; b[n] = 0.0
    return spsolve(ML.tocsr(), b)

# ---------- sizing helpers: 15–20 in x, ~10 in y ----------
def lp0_for_square(L, W, nx_segments=18, ny_segments=10):
    # square grid has spacing dx = 2*lp0; nx-1 = L/dx ≈ nx_segments
    lp0_x = L / (2.0 * nx_segments)
    lp0_y = W / (2.0 * ny_segments)
    return min(lp0_x, lp0_y)

def lp0_for_hex(L, W, nx_segments=18, ny_segments=10):
    # hex grid: dx = 2*lp0; dy = sqrt(3)*lp0; target nx-1 ≈ nx_segments, ny-1 ≈ ny_segments
    lp0_x = L / (2.0 * nx_segments)
    lp0_y = W / (np.sqrt(3.0) * ny_segments)
    return min(lp0_x, lp0_y)

# ---------- total-flow scaling (enforce 1000 vehicles) ----------
def total_outflow_from_set(phi, coords, efrom, eto, time_map, node_set):
    node_set = set(map(int, np.asarray(node_set).ravel()))
    Q = 0.0
    for u, v in zip(efrom, eto):
        if (u in node_set) ^ (v in node_set):  # exactly one in the set
            t = time_map[(u, v)]
            G = 1.0 / max(t, 1e-12)
            Q += G * (phi[u] - phi[v]) if (u in node_set) else G * (phi[v] - phi[u])
    return Q  # sign = out of the set

def scale_phi_to_total_flow(phi, coords, efrom, eto, time_map, node_set, Q_target):
    Q0 = total_outflow_from_set(phi, coords, efrom, eto, time_map, node_set)
    if abs(Q0) < 1e-18:
        return phi, 0.0
    alpha = Q_target / Q0
    return phi * alpha, alpha

# ---------- plot K_ij (mobility) ----------
def plot_mobility_tensors(coords, kxx, kxy, kyy, L, W, title_prefix='Mobility'):
    # Interpolate to a grid for clean panels
    ngx, ngy = 180, 140
    xg = np.linspace(0, L, ngx); yg = np.linspace(0, W, ngy)
    X, Y = np.meshgrid(xg, yg)
    Kxx = griddata(coords, kxx, (X, Y), method='linear')
    Kxy = griddata(coords, kxy, (X, Y), method='linear')
    Kyy = griddata(coords, kyy, (X, Y), method='linear')

    # Simple NaN-safe blur
    Kxx = nan_gaussian_blur(Kxx, 0.8)
    Kxy = nan_gaussian_blur(Kxy, 0.8)
    Kyy = nan_gaussian_blur(Kyy, 0.8)

    for Z, lab in [(Kxx, r'$K_{xx}$'), (Kxy, r'$K_{xy}$'), (Kyy, r'$K_{yy}$')]:
        plt.figure(figsize=(6.6, 4.8))
        plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
        plt.colorbar(label=lab)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x (m)'); plt.ylabel('y (m)')
        plt.title(f'{title_prefix}: {lab}')
        plt.tight_layout()

# -------------------------- path extraction --------------------------

def gradienttrace(gradx_interp,grady_interp,coords,src,dst,steplength=1e-4,
                             max_steps=7000, initial_angle_deg=10.0):
    """
    Follow continuous -grad(phi) by interpolating phi to a grid. Returns a snapped
    node path and the continuous polyline (Nx2).
    """
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())

    pt = coords[int(src)].astype(float)
    goal = coords[int(dst)].astype(float)
    path_xy = [pt.copy()]

    # First step: bias toward goal, optionally rotated
    if np.linalg.norm(goal - pt) > 1e-12:
        dir_goal = (goal - pt) / np.linalg.norm(goal - pt)
        if initial_angle_deg:
            ang = np.deg2rad(initial_angle_deg)
            ca, sa = np.cos(ang), np.sin(ang)
            dir_goal = np.array([ca*dir_goal[0] - sa*dir_goal[1],
                                 sa*dir_goal[0] + ca*dir_goal[1]], float)
        gx0 = gradx_interp((pt[1], pt[0]))
        gy0 = grady_interp((pt[1], pt[0]))
        gnrm = np.sqrt(gx0*gx0+gy0*gy0)
        if np.isfinite(gx0) and np.isfinite(gy0) and gnrm>1e-6:
            gvec0 = np.array([gx0, gy0], float)
            gnorm0 = np.linalg.norm(gvec0)
            if gnorm0 > 1e-15:
                gdir0 = gvec0 / gnorm0
                alpha = 0.0  # favor goal direction for the first step
                step_dir = alpha * (-gdir0) + (1 - alpha) * dir_goal
                step_dir /= max(np.linalg.norm(step_dir), 1e-15)
            else:
                step_dir = dir_goal
        else:
            step_dir = dir_goal
        pt = pt + steplength * step_dir
        pt[0] = np.clip(pt[0], x_min, x_max)
        pt[1] = np.clip(pt[1], y_min, y_max)
        path_xy.append(pt.copy())

    for index in range(int(max_steps)):
        gx = gradx_interp((pt[1], pt[0]))
        gy = grady_interp((pt[1], pt[0]))
        if not np.isfinite(gx) or not np.isfinite(gy):
            print("non-finite gradient, stopping")
            break
        gvec = np.array([gx, gy], float)
        gnrm = np.linalg.norm(gvec)
        gn = 0*gvec
        if gnrm < 1e-6:
            dirgoal = (goal - pt) / np.linalg.norm(goal - pt)
            step = steplength*dirgoal
            print("near zero grad, heading to goal")
            exit()
        else:
            gn = gvec/gnrm
            step = -steplength * gn
        pt = pt + step
        pt[0] = np.clip(pt[0], x_min, x_max)
        pt[1] = np.clip(pt[1], y_min, y_max)
        path_xy.append(pt.copy())
        print("index = ", index, "loc = ", pt," grd = ", gn,"gnrm =", gnrm)

        # if (index <=3900):
        #     print("index = ", index, "loc = ", pt," grd = ", gn,"gnrm =", gnrm)
        # else:
        #     break
        condition_to_goal = np.abs(pt[0]-goal[0])<=steplength and np.linalg.norm(pt-goal)<=1000*steplength
        if condition_to_goal:
            path_xy.append(goal.copy())
            break 
    return np.vstack(path_xy) if path_xy else np.zeros((0, 2))

def dijkstra_path(coords, efrom, eto, etime, src, dst):
    rows = np.concatenate([efrom, eto])
    cols = np.concatenate([eto, efrom])
    dat  = np.concatenate([etime, etime])
    N = coords.shape[0]
    adj = sparse.csr_matrix((dat,(rows,cols)), shape=(N,N))
    dist, preds = csgraph.dijkstra(adj, directed=False, indices=src, return_predecessors=True)
    if not np.isfinite(dist[dst]): return [], np.inf
    # backtrack
    path=[]; cur=dst
    while cur!=-9999 and cur!=src:
        path.append(cur); cur=preds[cur]
    path.append(src); path=path[::-1]
    return path, float(dist[dst])

def path_time(path, time_map):
    if len(path)<2:
        print("path_time: path too short")
        return np.nan
    tt=0.0
    #for a,b in zip(path[:-1], path[1:]):
    for i, (a, b) in enumerate(zip(path[:-1], path[1:]), start=1):
        t = time_map.get((a,b)) or time_map.get((b,a))
        if t is None:
            print("path_time: missing edge time for ", a, b, i, len(path))
            return np.nan
        tt+=t
    return tt

def path_time_from_mesh(path_coords, conductivity_grid, xg, yg, steplength=1e-4):
    """
    Compute travel time along a path using interpolated conductivity from a mesh grid.
    Integrates in substeps of steplength to handle spatially-varying conductivity.
    
    Args:
        path_coords: Nx2 array of (x, y) coordinates defining the path
        conductivity_grid: 2D mesh grid of conductivity values (inverse of time per unit length)
        xg: x-axis grid coordinates
        yg: y-axis grid coordinates
        steplength: step size for integration along path segments
    
    Returns:
        Total travel time along the path (sum of segment times)
    """
    from scipy.interpolate import RegularGridInterpolator
    
    if len(path_coords) < 2:
        print("path_time_from_mesh: path too short")
        return np.nan
    
    # Create interpolator for conductivity mesh
    # Note: conductivity_grid is indexed as [y, x] so we reverse xg, yg order
    cond_interp = RegularGridInterpolator((yg, xg), conductivity_grid, 
                                          bounds_error=False, fill_value=0.0)
    total_time = 0.0
    # Sum travel time over each path segment, integrating in steplength increments
    for i in range(len(path_coords) - 1):
        p_start = path_coords[i]
        p_end = path_coords[i + 1]
        segment_length = np.linalg.norm(p_end - p_start)
        if segment_length < 1e-12:
            continue
        # Number of substeps along this segment
        n_substeps = max(1, int(np.ceil(segment_length / steplength)))
        substep_length = segment_length / n_substeps
        # Integrate over substeps
        for j in range(n_substeps):
            # Position along segment (midpoint of substep)
            t_frac = (j + 0.5) / n_substeps
            pos = p_start + t_frac * (p_end - p_start)
            # Interpolate conductivity at this position
            # RegularGridInterpolator expects (y, x) order
            cond_value = cond_interp((pos[1], pos[0]))
            if not np.isfinite(cond_value) or cond_value <= 0:
                print(f"path_time_from_mesh: non-finite conductivity at segment {i}, substep {j}")
                return np.nan
            # Time = distance / speed, where speed ~ conductivity
            # Add time for this substep
            total_time += substep_length / cond_value
    return total_time

def time_map_to_conductivity_mesh(time_map, coords, efrom, eto, xg, yg):
    """
    Convert edge-based time_map to a continuous conductivity mesh grid.
    
    Args:
        time_map: Dictionary mapping (u,v) edges to travel times
        coords: Node coordinates (Nx2)
        efrom, eto: Edge connectivity arrays
        xg: x-axis grid coordinates
        yg: y-axis grid coordinates
    
    Returns:
        conductivity_grid: 2D mesh of conductivity values (1/time)
    """
    # Extract edge midpoints and conductivity values
    edge_midpts = 0.5 * (coords[efrom] + coords[eto])
    edge_times = np.array([time_map[(u, v)] for u, v in zip(efrom, eto)])
    edge_len = np.array([np.linalg.norm(coords[u] - coords[v]) for u, v in zip(efrom, eto)])
    edge_conductivity = edge_len / np.maximum(edge_times, 1e-12)
    # Create mesh grid and interpolate
    X, Y = np.meshgrid(xg, yg)
    conductivity_grid = griddata(edge_midpts, edge_conductivity, (X, Y), method='nearest')
    # Smooth with NaN-aware blur
    #conductivity_grid = nan_gaussian_blur(conductivity_grid, 0.8)
    return conductivity_grid

# -------------------------- plotting --------------------------

def plot_field_with_paths(X,Y,Phi,dPhidx,dPhidy, coords, src, dst,
                          greedy_coords=None, dijk_coords=None,
                          electrodes=None, hole_rect=None, crack_seg=None, title='',
                          show_arrows=True, arrow_stride=3):
    plt.figure(figsize=(7.5,6))
    im = plt.pcolormesh(X, Y, Phi, shading='auto', cmap='viridis')
    cbar = plt.colorbar(im, label='Potential V (arb.)')
    # add smooth equipotential contours
    try:
        CS = plt.contour(X, Y, Phi, levels=12, colors='k', linewidths=0.8, alpha=0.55)
        # optional labels on a few lines:
        # plt.clabel(CS, inline=True, fontsize=8, fmt='%.2f')
    except Exception:
        pass

    # optional arrows: point along -grad(phi) (steepest descent)
    if show_arrows and dPhidx is not None and dPhidy is not None:
        stride = max(1, int(arrow_stride))
        Xs = X[::stride, ::stride]; Ys = Y[::stride, ::stride]
        U = -dPhidx[::stride, ::stride]; V = -dPhidy[::stride, ::stride]
        mask = np.isfinite(Phi[::stride, ::stride]) & np.isfinite(U) & np.isfinite(V)
        if np.any(mask):
            plt.quiver(Xs[mask], Ys[mask], U[mask], V[mask],
                       color='white', angles='xy', scale_units='xy',
                       scale=20, width=0.0035, alpha=0.7, minlength=0.1)

    if greedy_coords is not None and len(greedy_coords)>0:
        plt.plot(greedy_coords[:,0], greedy_coords[:,1], 'r-', lw=0.5, label='Greedy steepest drop')
    if dijk_coords is not None and len(dijk_coords)>0:
        plt.plot(dijk_coords[:,0], dijk_coords[:,1], 'y-', lw=0.5, label='Dijkstra shortest time')
    plt.plot(coords[src,0], coords[src,1], 'go', ms=7, label='Source')
    plt.plot(coords[dst,0], coords[dst,1], 'ro', ms=7, label='Destination')
    ax = plt.gca()
    if electrodes:
        for (cx,cy,R,lab,col) in electrodes:
            ax.add_patch(Circle((cx,cy), R, fill=False, ec=col, lw=2.0, label=lab))
    if hole_rect is not None:
        cx,cy,w,h = hole_rect
        ax.add_patch(Rectangle((cx-w/2, cy-h/2), w,h, fill=False, ec='w', lw=0.5, ls='--', label='Hole'))
    if crack_seg is not None:
        a,b = np.asarray(crack_seg[0]).ravel()[:2], np.asarray(crack_seg[1]).ravel()[:2]
        plt.plot([a[0],b[0]],[a[1],b[1]], 'w--', lw=0.5, label='Crack')
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.title(title)
    plt.legend(loc='best'); plt.tight_layout()

    plt.tight_layout(); plt.savefig("tracemap.png", dpi=220)
    plt.close()
    print("Saved tracemap: tracemap.png")

# -------------------------- CASES --------------------------

def showconductance(coords,efrom,eto,M_jam,off,i,save_prefix,show):
    # Conductance visualization (no interpolation): scatter edge midpoints colored by G
    midpts = 0.5 * (coords[efrom] + coords[eto])
    conductance = np.array([-M_jam[u, v] for u, v in zip(efrom, eto)], dtype=float)
    print("minimum conductance : ", max(conductance))
    print("minimum conductance : ", min(conductance))
    plt.figure(figsize=(7.2, 5.6))
    plt.scatter(midpts[:,0], midpts[:,1], c=conductance, s=8, cmap='jet')
    plt.colorbar(label='Edge conductance (1/s)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Conductance after M-jam, offset {off:.2f}×lp0')
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.colorbar()
    cond_png = f"{save_prefix}_{i}_conductivity.png"
    plt.tight_layout(); plt.savefig(cond_png, dpi=220)
    if not show:
            plt.close()
    print("Saved conductivity field:", cond_png)


def showbjmesh(coords,bj,off,i,save_prefix,show,title,figname):
    # Plot bj as a mesh over node coordinates
    plt.figure(figsize=(7.2, 5.6))
    bx = coords[:,0]; by = coords[:,1]
    try:
            pcm = plt.tricontourf(bx, by, bj, levels=20, cmap='coolwarm')
            plt.colorbar(pcm, label='bj value')
    except Exception:
            plt.scatter(bx, by, c=bj, s=8, cmap='coolwarm')
            plt.colorbar(label='bj value')
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.title(f'bj RHS after Dirichlet (offset {off:.2f}×lp0)')
    plt.title(title)
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    bj_png = f"{save_prefix}_{i}_{figname}"
    plt.tight_layout(); plt.savefig(bj_png, dpi=220)
    if not show:
            plt.close()
    print("Saved bj plot:", figname)


def showphimesh(coords,phi,off,i,save_prefix,show,title,figname):
    plt.figure(figsize=(7.2, 5.6))
    plt.scatter(coords[:,0], coords[:,1], c=phi, s=8, cmap='viridis')
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.title(f'Phi field (offset {off:.2f}×lp0)')
    plt.title(title)
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.colorbar()
    phi_png = f"{save_prefix}_{i}_{figname}"
    plt.tight_layout(); plt.savefig(phi_png, dpi=220)
    if not show:
        plt.close()
    print("Saved Phi field:", phi_png)


def case_square_with_vertical_conductive_strip(save_prefix=None, show=False, seed=123):
    L = W = 2.0
    lp0_used = lp0
    coords, edges, lp0_used, elen = build_square_lattice(L=L, W=W, lp0=lp0_used)

    strip_xc   = L/2
    strip_w    = 0.10
    strip_y0   = 0.60   # bottom
    strip_y1   = 1.40   # top
    gain_gamma = 1e-6

    def in_strip(mx, my):
        return (abs(mx - strip_xc) <= strip_w/2) and (strip_y0 <= my <= strip_y1)

    mph_h, mph_v, mph_br, mph_bl = (20.0, 20.0, 20.0, 20.0)
    rows, cols, data = [], [], []
    efrom, eto, etime, K_edge = [], [], [], []
    time_map = {}

    for (u, v) in edges:
        p0, p1 = coords[u], coords[v]
        ori = edge_orientation(p0, p1)
        mph = mph_h if ori == 'horiz' else mph_v if ori == 'vert' else mph_br if ori == 'diag_br' else mph_bl
        speed = mph * MPH_TO_MPS
        el = np.linalg.norm(p1 - p0)
        t0 = el / max(speed, 1e-9)
        G0 = 1.0 / max(t0, 1e-12)
        # print("mph:", mph, " speed:", speed, " el:", el, " t0:", t0, " G0:", G0)
        # exit()

        mid = 0.5 * (p0 + p1)
        gain = gain_gamma if in_strip(mid[0], mid[1]) else 1.0
        G = G0 * gain; 
        t = t0 / gain

        rows += [u, v, u, v]; cols += [u, v, v, u]; data += [G, G, -G, -G]
        efrom.append(u); eto.append(v); etime.append(t); K_edge.append(G)
        time_map[(u, v)] = t
        time_map[(v, u)] = t

    M = sparse.csr_matrix((np.array(data, float),
                        (np.array(rows, int), np.array(cols, int))),
                        shape=(coords.shape[0], coords.shape[0]))
    efrom = np.array(efrom, int); eto = np.array(eto, int)
    etime = np.array(etime, float); K_edge = np.array(K_edge, float)

    tol = 1e-12
    left_nodes  = np.where(np.abs(coords[:, 0] - 0.0) <= tol)[0]
    right_nodes = np.where(np.abs(coords[:, 0] - L  ) <= tol)[0]
    ML, b  = apply_dirichlet_row_replace(M, left_nodes, 1.0)
    ML, b2 = apply_dirichlet_row_replace(ML, right_nodes, 0.0)
    b += b2

    phi = solve_with_component_pinning(ML, b, pinned_nodes=np.r_[left_nodes, right_nodes])

    # Scale to Q_total = 1000 vehicles through the left plate
    #phi, alpha = scale_phi_to_total_flow(phi, coords, efrom, eto, time_map, left_nodes, Q_target=1000.0)

    ng = 260
    xg = np.linspace(0, L, ng)
    yg = np.linspace(0, W, ng)
    X, Y = np.meshgrid(xg, yg)
    Phi = griddata(coords, phi, (X, Y), method='linear')
    # Phi = nan_gaussian_blur(Phi, 0.8)
    delx = xg[1]-xg[0]
    dely = yg[1]-yg[0]
    dPhidy, dPhidx = np.gradient(Phi, dely, delx, edge_order=2)
    gradx_interp = RegularGridInterpolator((yg, xg), dPhidx, method='quintic', bounds_error=False, fill_value=0.0)
    grady_interp = RegularGridInterpolator((yg, xg), dPhidy, method='quintic', bounds_error=False, fill_value=0.0)

    conductivity_grid = time_map_to_conductivity_mesh(time_map, coords, efrom, eto, xg, yg)
    src_phy = (0.0 + 10*lp0, W/2)
    dst_phy = (L - 10*lp0,   W/2)
    src = nearest_node(coords, src_phy)
    dst = nearest_node(coords, dst_phy)

    dijk_nodes, dijk_time = dijkstra_path(coords, efrom, eto, etime, src, dst)

    greedy_xy = gradienttrace(gradx_interp,grady_interp, coords, src, dst, steplength=steplen)
    greedy_time = path_time_from_mesh(greedy_xy, conductivity_grid, xg, yg, steplength=steplen)

    # greedy_time           = path_time(greedy_nodes, time_map)

    print(f'Dijkstra time: {dijk_time:.4f} s, Greedy time on mesh: {greedy_time:.4f} s')
    title = f'Square lattice with vertical conductive strip'
    hole_rect = (strip_xc, 0.5*(strip_y0+strip_y1), strip_w, (strip_y1-strip_y0))  # visual only

    plot_field_with_paths(
        X, Y, Phi, dPhidx, dPhidy, coords, src, dst,
        greedy_coords=greedy_xy,
        dijk_coords=coords[dijk_nodes] if dijk_nodes else None,
        electrodes=None,
        hole_rect=hole_rect,
        title=title)

if __name__ == "__main__":
    case_square_with_vertical_conductive_strip(save_prefix=None, show=False, seed=44)
    plt.show()
