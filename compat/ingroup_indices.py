"""Equivalent SST within-group ordinal indexing; preserves tensor computation."""
import torch
@torch.no_grad()
def forward(group_inds,out_inds):
    if group_inds.numel()==0: return
    order=torch.argsort(group_inds,stable=True)
    groups=group_inds[order]
    positions=torch.arange(group_inds.numel(),device=group_inds.device,dtype=group_inds.dtype)
    boundary=torch.ones_like(groups,dtype=torch.bool)
    boundary[1:]=groups[1:]!=groups[:-1]
    starts=torch.where(boundary,positions,torch.zeros_like(positions)).cummax(0).values
    out_inds[order]=positions-starts
