"""
data/reference_builder.py

Key additions vs previous version
───────────────────────────────────
• Stores ALL sample landmarks and angle matrices (not just the medoid).
  This lets main_window select the pro video most similar to the user
  rather than always using the same medoid.

• find_best_match_from_ref(user_angles, ref) — called by ProcessingThread
  to select the best-matching pro index.

• Backward compatible: old .npz files (without all_landmarks_3d) still
  load cleanly; the matcher falls back to medoid index 0.

• max_frames = 250 prevents serve videos from hanging for 15+ minutes.
• Per-video 90-second wall-clock timeout via threading.Timer.
"""

import os
import time
import threading
import numpy as np
from dtaidistance import dtw, dtw_barycenter

from processing.pose_extractor import PoseExtractor

VIDEO_EXTS = {'.mp4','.avi','.mov','.mkv','.wmv','.m4v'}
_REF_MAX_FRAMES = 250     # frames per reference video (250 @ 30fps = 8.3s)
_VIDEO_TIMEOUT_S = 90     # per-video wall-clock timeout


def list_video_files(folder):
    if not os.path.isdir(folder): return []
    return [os.path.join(folder,name)
            for name in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder,name))
            and os.path.splitext(name)[1].lower() in VIDEO_EXTS]


def reference_path(base_prof_dir,stroke):
    return os.path.join(base_prof_dir,stroke,'reference.npz')


def _resample_1d(arr,n):
    arr=np.asarray(arr,dtype=np.float32)
    if len(arr)==0: return np.zeros(n,dtype=np.float32)
    if len(arr)==1: return np.full(n,float(arr[0]),dtype=np.float32)
    return np.interp(np.linspace(0,1,n),np.linspace(0,1,len(arr)),arr).astype(np.float32)


def _resample_landmarks(arr,n):
    arr=np.asarray(arr,dtype=np.float32)
    if arr.size==0: return np.zeros((n,33,3),dtype=np.float32)
    t,num_lm,dims=arr.shape; out=np.zeros((n,num_lm,dims),dtype=np.float32)
    for lm in range(num_lm):
        for d in range(dims):
            out[:,lm,d]=_resample_1d(arr[:,lm,d],n)
    return out


def _barycenter_1d(seqs):
    seqs=[np.asarray(s,dtype=np.double) for s in seqs if len(s)>0]
    if not seqs: return np.zeros(120,dtype=np.float32)
    if len(seqs)==1: return seqs[0].astype(np.float32)
    init=np.mean(np.vstack(seqs),axis=0)
    bary=dtw_barycenter.dba_loop(
        seqs,c=np.asarray(init,dtype=np.double),
        max_it=10,thr=1e-3,use_c=False)
    return np.asarray(bary,dtype=np.float32)


def _extract_sample_with_timeout(video_path,extractor,target_len,timeout_s):
    result=[None]; exc=[None]; finished=threading.Event()
    def _worker():
        try:
            data=extractor.extract_from_video(video_path,max_frames=_REF_MAX_FRAMES)
            if 'landmarks_3d' not in data or len(data['landmarks_3d'])<15:
                result[0]=None; return
            angles=extractor.compute_joint_angles(data['landmarks_3d'])
            angles_r={k:_resample_1d(v,target_len) for k,v in angles.items()}
            result[0]={
                'video_path':    video_path,
                'angles_raw':    angles,
                'angles_r':      angles_r,
                'landmarks_3d_r':_resample_landmarks(data['landmarks_3d'],target_len),
                'landmarks_2d_r':_resample_landmarks(data['landmarks_2d'],target_len),
                'fps':           float(data.get('fps',30.0)),
                'target_len':    target_len,
            }
        except Exception as e:
            exc[0]=e
        finally:
            finished.set()
    t=threading.Thread(target=_worker,daemon=True); t.start()
    if not finished.wait(timeout=timeout_s):
        return None,f"timed out after {timeout_s}s"
    if exc[0] is not None:
        return None,str(exc[0])
    return result[0],None


def _build_reference_from_samples(samples,stroke,target_len=120):
    joint_names=sorted(samples[0]['angles_r'].keys())
    ideal_angles={j:_barycenter_1d([s['angles_r'][j] for s in samples])
                  for j in joint_names}
    best_idx,best_score=0,float('inf')
    for i,sample in enumerate(samples):
        score=sum(dtw.distance_fast(sample['angles_r'][j].astype(np.double),
                                     ideal_angles[j].astype(np.double))
                  for j in joint_names)
        if score<best_score: best_score,best_idx=score,i
    medoid=samples[best_idx]
    return {
        'stroke':        stroke,
        'target_len':    target_len,
        'num_videos':    len(samples),
        'source_videos': [s['video_path'] for s in samples],
        'joint_names':   joint_names,
        'ideal_angles':  ideal_angles,
        'medoid_video':  medoid['video_path'],
        'medoid_data': {
            'landmarks_3d':  medoid['landmarks_3d_r'],
            'landmarks_2d':  medoid['landmarks_2d_r'],
            'frame_indices': np.arange(target_len,dtype=np.int32),
            'fps':           medoid['fps'],
            'total_frames':  target_len,
        },
        # All samples for best-match selection
        '_samples': samples,
    }


def save_reference(reference,base_prof_dir,stroke):
    ref_path=reference_path(base_prof_dir,stroke)
    os.makedirs(os.path.dirname(ref_path),exist_ok=True)
    jnames=reference['joint_names']
    samples=reference.get('_samples',[])
    n_vids=len(samples) if samples else 1

    # All-samples angle matrix: shape (n_vids, n_joints, target_len)
    if samples:
        all_angle_matrix=np.stack([
            np.stack([s['angles_r'][j] for j in jnames])
            for s in samples]).astype(np.float32)
        all_lm3d=np.stack([s['landmarks_3d_r'] for s in samples]).astype(np.float32)
        all_lm2d=np.stack([s['landmarks_2d_r'] for s in samples]).astype(np.float32)
        all_fps=np.array([s['fps'] for s in samples],dtype=np.float32)
    else:
        tl=reference['target_len']
        all_angle_matrix=np.zeros((1,len(jnames),tl),dtype=np.float32)
        all_lm3d=reference['medoid_data']['landmarks_3d'][np.newaxis]
        all_lm2d=reference['medoid_data']['landmarks_2d'][np.newaxis]
        all_fps=np.array([reference['medoid_data']['fps']],dtype=np.float32)

    np.savez_compressed(
        ref_path,
        stroke               = np.array(reference['stroke']),
        target_len           = np.array(reference['target_len'],  dtype=np.int32),
        num_videos           = np.array(reference['num_videos'],   dtype=np.int32),
        source_videos        = np.array(reference['source_videos'],dtype=object),
        joint_names          = np.array(jnames,                    dtype=object),
        ideal_angle_matrix   = np.stack([reference['ideal_angles'][j]
                                          for j in jnames]).astype(np.float32),
        medoid_video         = np.array(reference['medoid_video']),
        medoid_landmarks_3d  = reference['medoid_data']['landmarks_3d'],
        medoid_landmarks_2d  = reference['medoid_data']['landmarks_2d'],
        medoid_frame_indices = reference['medoid_data']['frame_indices'],
        medoid_fps           = np.array(reference['medoid_data']['fps'],dtype=np.float32),
        medoid_total_frames  = np.array(reference['medoid_data']['total_frames'],dtype=np.int32),
        # New: all samples
        all_angle_matrix     = all_angle_matrix,
        all_landmarks_3d     = all_lm3d,
        all_landmarks_2d     = all_lm2d,
        all_fps              = all_fps,
    )
    return ref_path


def load_reference(base_prof_dir,stroke):
    ref_path=reference_path(base_prof_dir,stroke)
    if not os.path.exists(ref_path): raise FileNotFoundError(ref_path)
    z=np.load(ref_path,allow_pickle=True)
    joint_names=[str(x) for x in z['joint_names'].tolist()]
    ideal_mat=z['ideal_angle_matrix']
    ref={
        'stroke':        str(z['stroke'].item()),
        'target_len':    int(z['target_len'].item()),
        'num_videos':    int(z['num_videos'].item()),
        'source_videos': [str(x) for x in z['source_videos'].tolist()],
        'joint_names':   joint_names,
        'ideal_angles':  {n:ideal_mat[i].astype(np.float32)
                          for i,n in enumerate(joint_names)},
        'medoid_video':  str(z['medoid_video'].item()),
        'medoid_data': {
            'landmarks_3d':  z['medoid_landmarks_3d'].astype(np.float32),
            'landmarks_2d':  z['medoid_landmarks_2d'].astype(np.float32),
            'frame_indices': z['medoid_frame_indices'].astype(np.int32),
            'fps':           float(z['medoid_fps'].item()),
            'total_frames':  int(z['medoid_total_frames'].item()),
        },
    }
    # Load all-samples data if present (backward compat)
    if 'all_landmarks_3d' in z:
        ref['all_angle_matrix'] = z['all_angle_matrix'].astype(np.float32)
        ref['all_landmarks_3d'] = z['all_landmarks_3d'].astype(np.float32)
        ref['all_landmarks_2d'] = z['all_landmarks_2d'].astype(np.float32)
        ref['all_fps']          = z['all_fps'].astype(np.float32)
    return ref


def find_best_match_from_ref(user_angles, ref):
    """
    Return the index (0-based) of the pro video whose joint angle sequence
    is closest to user_angles (dict {joint_name: 1-D array}).

    Uses mean-squared-error over all joints on resampled sequences.
    Falls back to 0 (medoid) if all-samples data is not available.
    """
    if 'all_angle_matrix' not in ref:
        return 0
    all_m = ref['all_angle_matrix']   # (n_vids, n_joints, target_len)
    jnames = ref['joint_names']
    target_len = all_m.shape[2]
    n_vids = all_m.shape[0]
    scores = np.zeros(n_vids, dtype=np.float64)
    for ji, jname in enumerate(jnames):
        if jname not in user_angles:
            continue
        u_raw = np.asarray(user_angles[jname], dtype=np.float32)
        u_res = np.interp(np.linspace(0,1,target_len),
                          np.linspace(0,1,max(len(u_raw),1)), u_raw)
        for vi in range(n_vids):
            scores[vi] += float(np.mean((u_res - all_m[vi,ji,:])**2))
    return int(np.argmin(scores))


def reference_is_stale(base_prof_dir,stroke):
    videos=list_video_files(os.path.join(base_prof_dir,stroke))
    ref_path=reference_path(base_prof_dir,stroke)
    if not videos: return False
    if not os.path.exists(ref_path): return True
    return max(os.path.getmtime(v) for v in videos)>os.path.getmtime(ref_path)


def build_reference_for_stroke(base_prof_dir,stroke,target_len=120,verbose=True):
    videos=list_video_files(os.path.join(base_prof_dir,stroke))
    if not videos:
        raise FileNotFoundError(f'No videos in {os.path.join(base_prof_dir,stroke)}')
    extractor=PoseExtractor(); samples=[]
    for idx,vp in enumerate(videos,start=1):
        name=os.path.basename(vp)
        if verbose: print(f'[{stroke}] ({idx}/{len(videos)}) Processing: {name}',flush=True)
        t0=time.time()
        sample,err=_extract_sample_with_timeout(vp,extractor,target_len,_VIDEO_TIMEOUT_S)
        elapsed=time.time()-t0
        if err:
            print(f'[{stroke}]   Skipped {name}: {err}',flush=True)
        elif sample is None:
            print(f'[{stroke}]   Skipped {name}: no valid pose data',flush=True)
        else:
            samples.append(sample)
            if verbose: print(f'[{stroke}]   OK ({elapsed:.1f}s)',flush=True)
    if not samples:
        raise RuntimeError(f'No valid pose data for stroke: {stroke}')
    ref=_build_reference_from_samples(samples,stroke,target_len=target_len)
    save_reference(ref,base_prof_dir,stroke)
    if verbose:
        med=os.path.basename(ref['medoid_video'])
        print(f'[{stroke}] Built from {ref["num_videos"]} videos. Medoid: {med}')
    return ref


def load_or_build_reference(base_prof_dir,stroke,target_len=120,force=False,verbose=False):
    if force or reference_is_stale(base_prof_dir,stroke):
        return build_reference_for_stroke(base_prof_dir,stroke,
                                          target_len=target_len,verbose=verbose)
    try:
        return load_reference(base_prof_dir,stroke)
    except FileNotFoundError:
        return build_reference_for_stroke(base_prof_dir,stroke,
                                          target_len=target_len,verbose=verbose)


def build_all_references(base_prof_dir,strokes=('forehand','backhand','serve'),
                         target_len=120,force=False):
    built={}
    for stroke in strokes:
        if not list_video_files(os.path.join(base_prof_dir,stroke)): continue
        built[stroke]=load_or_build_reference(base_prof_dir,stroke,
                                               target_len=target_len,
                                               force=force,verbose=True)
    return built