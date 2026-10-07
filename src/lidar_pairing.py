"""Select real paired scan timestamps for which odometry TF already exists."""

def pair_indices(stamps0,stamps1,tf_latest_ns,now_ns,used=None):
    candidates=[]
    for i,a in enumerate(stamps0):
        for j,b in enumerate(stamps1):
            target=max(a,b)
            if (target<=tf_latest_ns and -20_000_000<=now_ns-target<400_000_000 and
                    abs(a-b)<=80_000_000 and (a,b)!=used):
                candidates.append((target,-abs(a-b),i,j))
    if not candidates:return None
    _,_,i,j=max(candidates)
    return i,j
