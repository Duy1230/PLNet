import torch
import numpy as np
from torch.utils.data.dataloader import default_collate

from hawp.base import _C


def _encode_ls_fallback(lines, input_height, input_width, height, width, num_lines):
    """
    Pure-torch fallback for line-segment encoding.

    This is used when the CUDA extension is not available or cannot run
    for the current tensor/device.
    """
    device = lines.device
    dtype = lines.dtype

    map_out = torch.zeros((6, height, width), device=device, dtype=dtype)
    tmap = torch.zeros((1, height, width), device=device, dtype=dtype)
    label = torch.zeros((num_lines, height, width), device=device, dtype=torch.bool)

    if num_lines <= 0:
        return map_out, label, tmap

    # Scale lines from input resolution to target resolution.
    xs = float(width) / float(input_width)
    ys = float(height) / float(input_height)

    x1 = lines[:, 0] * xs
    y1 = lines[:, 1] * ys
    x2 = lines[:, 2] * xs
    y2 = lines[:, 3] * ys
    dx = x2 - x1
    dy = y2 - y1
    norm2 = dx * dx + dy * dy

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    px = xx.reshape(-1)
    py = yy.reshape(-1)
    n_pix = px.numel()

    inf = torch.full((n_pix,), float("inf"), device=device, dtype=dtype)
    min_dis = inf.clone()
    best_line_idx = torch.full((n_pix,), -1, device=device, dtype=torch.long)
    best_flag = torch.zeros((n_pix,), device=device, dtype=torch.bool)
    best_t = torch.zeros((n_pix,), device=device, dtype=dtype)

    best_ax = torch.zeros((n_pix,), device=device, dtype=dtype)
    best_ay = torch.zeros((n_pix,), device=device, dtype=dtype)
    best_ux = torch.zeros((n_pix,), device=device, dtype=dtype)
    best_uy = torch.zeros((n_pix,), device=device, dtype=dtype)
    best_vx = torch.zeros((n_pix,), device=device, dtype=dtype)
    best_vy = torch.zeros((n_pix,), device=device, dtype=dtype)

    # Chunk to avoid excessive memory when the number of lines is large.
    chunk_size = 256
    for start in range(0, num_lines, chunk_size):
        end = min(start + chunk_size, num_lines)
        x1c = x1[start:end]
        y1c = y1[start:end]
        x2c = x2[start:end]
        y2c = y2[start:end]
        dxc = dx[start:end]
        dyc = dy[start:end]
        norm2c = norm2[start:end]

        # [C, HW]
        t = (
            (px.unsqueeze(0) - x1c.unsqueeze(1)) * dxc.unsqueeze(1)
            + (py.unsqueeze(0) - y1c.unsqueeze(1)) * dyc.unsqueeze(1)
        ) / (norm2c.unsqueeze(1) + 1e-6)
        flag = (t <= 1.0) & (t >= 0.0)
        t_clamped = t.clamp(0.0, 1.0)

        ax = x1c.unsqueeze(1) + t_clamped * dxc.unsqueeze(1) - px.unsqueeze(0)
        ay = y1c.unsqueeze(1) + t_clamped * dyc.unsqueeze(1) - py.unsqueeze(0)
        dis = ax * ax + ay * ay

        ux = x1c.unsqueeze(1) - px.unsqueeze(0)
        uy = y1c.unsqueeze(1) - py.unsqueeze(0)
        vx = x2c.unsqueeze(1) - px.unsqueeze(0)
        vy = y2c.unsqueeze(1) - py.unsqueeze(0)

        norm_u2 = ux * ux + uy * uy
        norm_v2 = vx * vx + vy * vy
        use_u_first = norm_u2 < norm_v2

        chunk_min_dis, chunk_idx = dis.min(dim=0)
        update_mask = chunk_min_dis < min_dis
        if not torch.any(update_mask):
            continue

        cols = torch.nonzero(update_mask, as_tuple=False).flatten()
        rows = chunk_idx[cols]

        min_dis[cols] = chunk_min_dis[cols]
        best_line_idx[cols] = rows + start
        best_flag[cols] = flag[rows, cols]
        best_t[cols] = t_clamped[rows, cols]

        best_ax[cols] = ax[rows, cols]
        best_ay[cols] = ay[rows, cols]

        use_u = use_u_first[rows, cols]
        best_ux[cols] = torch.where(use_u, ux[rows, cols], vx[rows, cols])
        best_uy[cols] = torch.where(use_u, uy[rows, cols], vy[rows, cols])
        best_vx[cols] = torch.where(use_u, vx[rows, cols], ux[rows, cols])
        best_vy[cols] = torch.where(use_u, vy[rows, cols], uy[rows, cols])

    map_out[0] = best_ax.view(height, width)
    map_out[1] = best_ay.view(height, width)
    map_out[2] = best_ux.view(height, width)
    map_out[3] = best_uy.view(height, width)
    map_out[4] = best_vx.view(height, width)
    map_out[5] = best_vy.view(height, width)
    tmap[0] = best_t.view(height, width)

    valid = best_line_idx >= 0
    if torch.any(valid):
        pix_idx = torch.nonzero(valid, as_tuple=False).flatten()
        line_idx = best_line_idx[pix_idx]
        label.view(num_lines, -1)[line_idx, pix_idx] = best_flag[pix_idx]

    return map_out, label, tmap


def _encode_ls(lines, input_height, input_width, height, width, num_lines):
    # Fast path: compiled CUDA op.
    if _C is not None and hasattr(_C, "encodels"):
        try:
            return _C.encodels(lines, input_height, input_width, height, width, num_lines)
        except Exception:
            # Fall back silently to keep training running on environments
            # where extension loading succeeds but invocation fails.
            pass
    return _encode_ls_fallback(lines, input_height, input_width, height, width, num_lines)


class HAFMencoder(object):
    def __init__(self, cfg):
        self.dis_th = cfg.ENCODER.DIS_TH
        self.ang_th = cfg.ENCODER.ANG_TH
        self.num_static_pos_lines = cfg.ENCODER.NUM_STATIC_POS_LINES
        self.num_static_neg_lines = cfg.ENCODER.NUM_STATIC_NEG_LINES
        self.bck_weight = cfg.ENCODER.BACKGROUND_WEIGHT
    def __call__(self,annotations):
        targets = []
        metas   = []
        for ann in annotations:
            t,m = self._process_per_image(ann)
            targets.append(t)
            metas.append(m)
        
        return default_collate(targets),metas

    def adjacent_matrix(self, n, edges, device):
        mat = torch.zeros(n+1,n+1,dtype=torch.bool,device=device)
        if edges.size(0)>0:
            mat[edges[:,0], edges[:,1]] = 1
            mat[edges[:,1], edges[:,0]] = 1
        return mat

    def _process_per_image(self,ann):
        junctions = ann['junctions']
        device = junctions.device
        height, width = ann['height'], ann['width']
        # jmap = torch.zeros((height,width),device=device)
        # joff = torch.zeros((2,height,width),device=device,dtype=torch.float32)
        jmap = np.zeros((height,width),dtype=np.float32)
        joff = np.zeros((2,height,width),dtype=np.float32)
        # junctions[:,0] = junctions[:,0].clamp(min=0,max=width-1)
        # junctions[:,1] = junctions[:,1].clamp(min=0,max=height-1)
        junctions_np = junctions.cpu().numpy()
        xint, yint = junctions_np[:,0].astype(np.int32), junctions_np[:,1].astype(np.int32)
        off_x = junctions_np[:,0] - np.floor(junctions_np[:,0]) - 0.5
        off_y = junctions_np[:,1] - np.floor(junctions_np[:,1]) - 0.5
        jmap[yint, xint] = 1
        joff[0,yint, xint] = off_x
        joff[1,yint, xint] = off_y
        # xint,yint = junctions[:,0].long(), junctions[:,1].long()
        # off_x = junctions[:,0] - xint.float()-0.5
        # off_y = junctions[:,1] - yint.float()-0.5

        # jmap[yint,xint] = 1
        # joff[0,yint,xint] = off_x
        # joff[1,yint,xint] = off_y
        jmap = torch.from_numpy(jmap).to(device)
        joff = torch.from_numpy(joff).to(device)

        edges_positive = ann['edges_positive']
        edges_negative = ann['edges_negative']
        
        pos_mat = self.adjacent_matrix(junctions.size(0),edges_positive,device)
        neg_mat = self.adjacent_matrix(junctions.size(0),edges_negative,device)        
        lines = torch.cat((junctions[edges_positive[:,0]], junctions[edges_positive[:,1]]),dim=-1)
        lines_neg = torch.cat((junctions[edges_negative[:2000,0]],junctions[edges_negative[:2000,1]]),dim=-1)
        lmap, _, _ = _encode_ls(lines, height, width, height, width, lines.size(0))

        center_points = (lines[:,:2] + lines[:,2:])/2.0
        cmap = torch.zeros((height,width),device=device)
        xx, yy =torch.meshgrid(torch.arange(width,dtype=torch.float32,device=device),torch.arange(height,dtype=torch.float32,device=device),indexing='xy')

        ctl_dis = torch.min((xx[...,None]-center_points[None,None,:,0])**2 + (yy[...,None]-center_points[None,None,:,1])**2,dim=-1)[0]
        cmask = ctl_dis<=4.0

        cxint, cyint = center_points[:,0].long(), center_points[:,1].long()
        cmap[cyint,cxint] = 1


        lpos = np.random.permutation(lines.cpu().numpy())[:self.num_static_pos_lines]
        lneg = np.random.permutation(lines_neg.cpu().numpy())[:self.num_static_neg_lines]
        # lpos = lines[torch.randperm(lines.size(0),device=device)][:self.num_static_pos_lines]
        # lneg = lines_neg[torch.randperm(lines_neg.size(0),device=device)][:self.num_static_neg_lines]
        lpos = torch.from_numpy(lpos).to(device)
        lneg = torch.from_numpy(lneg).to(device)
        
        lpre = torch.cat((lpos,lneg),dim=0)
        _swap = (torch.rand(lpre.size(0))>0.5).to(device)
        lpre[_swap] = lpre[_swap][:,[2,3,0,1]]
        lpre_label = torch.cat(
            [
                torch.ones(lpos.size(0),device=device),
                torch.zeros(lneg.size(0),device=device)
             ])

        meta = {
            'junc': junctions,
            'Lpos':   pos_mat,
            'Lneg':   neg_mat,
            'lpre':      lpre,
            'lpre_label': lpre_label,
            'lines':     lines,
        }


        dismap = torch.sqrt(lmap[0]**2+lmap[1]**2)[None]
        def _normalize(inp):
            mag = torch.sqrt(inp[0]*inp[0]+inp[1]*inp[1])
            return inp/(mag+1e-6)

        md_map = _normalize(lmap[:2])
        st_map = _normalize(lmap[2:4])
        ed_map = _normalize(lmap[4:])

        md_ = md_map.reshape(2,-1).t()
        st_ = st_map.reshape(2,-1).t()
        ed_ = ed_map.reshape(2,-1).t()
        Rt = torch.cat(
                (torch.cat((md_[:,None,None,0],md_[:,None,None,1]),dim=2),
                 torch.cat((-md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)
        R = torch.cat(
                (torch.cat((md_[:,None,None,0], -md_[:,None,None,1]),dim=2),
                 torch.cat((md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)

        Rtst_ = torch.matmul(Rt, st_[:,:,None]).squeeze(-1).t()
        Rted_ = torch.matmul(Rt, ed_[:,:,None]).squeeze(-1).t()
        swap_mask = (Rtst_[1]<0)*(Rted_[1]>0)
        pos_ = Rtst_.clone()
        neg_ = Rted_.clone()
        temp = pos_[:,swap_mask]
        pos_[:,swap_mask] = neg_[:,swap_mask]
        neg_[:,swap_mask] = temp

        pos_[0] = pos_[0]#.clamp(min=1e-9)
        pos_[1] = pos_[1]#.clamp(min=1e-9)
        neg_[0] = neg_[0]#.clamp(min=1e-9)
        neg_[1] = neg_[1]#.clamp(max=-1e-9)
        
        mask = (dismap.view(-1)<=self.dis_th).float()

        pos_map = pos_.reshape(-1,height,width)
        neg_map = neg_.reshape(-1,height,width)

        md_angle  = torch.atan2(md_map[1], md_map[0])
        pos_angle = torch.atan2(pos_map[1],pos_map[0])
        neg_angle = torch.atan2(neg_map[1],neg_map[0])
        mask *= (pos_angle.reshape(-1)>self.ang_th*np.pi/2.0)
        mask *= (neg_angle.reshape(-1)<-self.ang_th*np.pi/2.0)

        pos_angle_n = pos_angle/(np.pi/2)
        neg_angle_n = -neg_angle/(np.pi/2)
        md_angle_n  = md_angle/(np.pi*2) + 0.5
        mask    = mask.reshape(height,width)

        mask[mask<1e-3] = self.bck_weight
        # import pdb; pdb.set_trace()
        hafm_ang = torch.cat((md_angle_n[None],pos_angle_n[None],neg_angle_n[None],),dim=0)
        hafm_dis   = dismap.clamp(max=self.dis_th)/self.dis_th
        mask = mask[None]
        target = {'jloc':jmap[None],
                'joff':joff,
                'cloc': cmap[None],
                'md': hafm_ang,
                'dis': hafm_dis,
                'mask': mask
               }
        return target, meta