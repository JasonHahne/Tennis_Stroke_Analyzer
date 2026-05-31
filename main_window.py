"""Tennis Stroke Analyzer – main_window.py"""
import os,re,json
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QLabel,
    QComboBox,QFileDialog,QTextEdit,QProgressBar,QSlider,QTabWidget,
    QGroupBox,QMessageBox,QScrollArea,
)
from PyQt5.QtCore import Qt,QThread,pyqtSignal,QTimer
from PyQt5.QtGui import QImage,QPixmap
import matplotlib; matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from processing.pose_extractor import SKELETON_CONNECTIONS,KEY_LANDMARKS
from data.reference_builder import load_reference,load_or_build_reference

DARK='#1a1a2e';PANEL='#16213e';ACCENT='#0f3460'
RED='#e94560';BLUE='#00ccff';PRO_COLOR='#ff7700'
GREEN='#00ff88';YELLOW='#ffe033';AXIS_C='#888888';INFO_C='#4499ff'
APP_STYLE=f"""
QMainWindow,QWidget{{background-color:{DARK};color:white;}}
QGroupBox{{border:1px solid {ACCENT};border-radius:6px;margin-top:12px;padding-top:10px;}}
QGroupBox::title{{color:{BLUE};subcontrol-position:top left;padding:0 4px;left:8px;}}
QPushButton{{background:{PANEL};border:2px solid {ACCENT};border-radius:6px;padding:8px 14px;color:white;font-size:13px;}}
QPushButton:hover{{background:{ACCENT};}}QPushButton:pressed{{background:{RED};}}
QPushButton:disabled{{background:#333;color:#666;}}
QComboBox{{background:{PANEL};border:1px solid {ACCENT};color:white;padding:4px;border-radius:4px;font-size:11px;}}
QComboBox QAbstractItemView{{background:{PANEL};color:white;font-size:11px;}}
QTabWidget::pane{{border:1px solid {ACCENT};}}
QTabBar::tab{{background:{PANEL};color:white;padding:8px 14px;border:1px solid {ACCENT};}}
QTabBar::tab:selected{{background:{ACCENT};}}
QTextEdit{{background:{PANEL};color:{GREEN};border:1px solid {ACCENT};font-family:monospace;font-size:12px;}}
QProgressBar{{background:{PANEL};border:1px solid {ACCENT};border-radius:4px;text-align:center;color:white;}}
QProgressBar::chunk{{background:{RED};border-radius:4px;}}
QSlider::groove:horizontal{{background:{PANEL};height:6px;border-radius:3px;}}
QSlider::handle:horizontal{{background:{RED};width:14px;height:14px;margin:-4px 0;border-radius:7px;}}
QLabel{{color:white;}}QScrollArea{{border:none;}}
"""
RIGHT_ARM={12,14,16};LEFT_ARM={11,13,15}
STROKE_PHASES={
'forehand':[('split_step','Split Step','#4488ff'),('unit_turn','Unit Turn','#44ff88'),
            ('racket_drop','Racket Drop','#ff8844'),('acceleration','Accel.','#ff4488'),
            ('contact','Contact','#ffff44'),('follow_through','Follow-Thru','#cc44ff')],
'backhand':[('split_step','Split Step','#4488ff'),('unit_turn','Unit Turn','#44ff88'),
            ('takeback','Takeback','#ff8844'),('racket_drop','Racket Drop','#ff4422'),
            ('contact','Contact','#ffff44'),('follow_through','Follow-Thru','#cc44ff')],
'serve':   [('stance','Stance/Toss','#4488ff'),('trophy','Trophy','#44ff88'),
            ('racket_drop','Racket Drop','#ff8844'),('acceleration','Accel.','#ff4488'),
            ('contact','Contact','#ffff44'),('follow_through','Follow-Thru','#cc44ff')],
}
JOINT_DESCRIPTIONS={
'right_elbow':("Hitting Elbow Angle","Measures how bent your hitting arm is.\nAt contact for forehand: ~160-175 deg (nearly straight).\nLESS than pro = arm too bent = less reach & power.\nMORE than pro = risk of hyperextension."),
'left_elbow':("Non-Dominant Elbow","Your balance arm's bend.\nForehand: tuck it in for rotation. Two-hand BH: both are hitting.\nLESS = restricted rotation. MORE = arm flailing."),
'right_shoulder':("Hitting Shoulder","Shoulder rotation through the swing.\nAt contact: ~45 deg open toward target.\nLESS = not rotating through = pushed/weak shot.\nMORE = over-rotating = loss of control."),
'left_shoulder':("Non-Dominant Shoulder","Should stay closed until contact, then open.\nLESS = poor unit turn. MORE = opening too early = power loss."),
'hip_rotation':("Hip Rotation","Hips should lead shoulders for the kinetic chain.\nLESS = hitting with arm only = power ceiling.\nMORE = hips spinning out = off-balance finish."),
'right_knee':("Hitting-Side Knee","Provides the platform to push up into the shot.\nLESS = no leg drive = flat footwork.\nMORE = excessive crouch = may cause timing issues."),
'left_knee':("Support-Side Knee","Balance leg mirrors the hitting knee.\nLESS = stiff support leg = blocked rotation.\nMORE = lunging too far = unstable base."),
}
EXPECTED_STROKE_DURATION_S={'forehand':0.85,'backhand':0.90,'serve':1.20}
SPEED_OPTIONS=[("Auto-detect",None),("1x real-time",1.0),("0.5x (2x slow)",0.5),
               ("0.25x (4x slow)",0.25),("0.125x (8x slow)",0.125),
               ("0.1x (10x slow)",0.10),("0.05x (20x slow)",0.05)]
COURT_LEN=23.77;DBL_W=10.97;SGL_W=8.23
NET_X=COURT_LEN/2;NET_H_CTR=0.914;NET_H_POST=1.07
SVC_DEPTH=6.40;SVC_LINE_P=NET_X-SVC_DEPTH;SVC_LINE_O=NET_X+SVC_DEPTH
SW_METERS=0.44
PANEL_INFO={
'speed':"Racket Head Speed\n\nEstimated HEAD speed over real time (slow-mo corrected).\nDashed lines=phase boundaries | Yellow=contact moment.",
'phases':"Phase Durations\n\nTime in each stroke phase (real seconds) vs pro.\n^ = you take longer  |  v = you're faster",
'contact':"Contact Point -- Top View\n\nWhere racket head was relative to your body at impact.\nGood zone = arm extended in front.",
'metrics':"Key Metrics\n\nRacket speed, estimated ball exit speed, phase comparison.",
}

def parse_pro_name(video_path):
    """'alcaraz_forehand1.mp4' -> 'Alcaraz'"""
    if not video_path: return "Pro"
    base=os.path.basename(video_path); name_part=base.split('_')[0]
    return name_part.capitalize() if name_part else "Pro"

def normalise_world_seq(world_seq):
    seq=world_seq.copy(); hips=(seq[:,23]+seq[:,24])/2.0; seq-=hips[:,None,:]
    sw=np.linalg.norm(seq[:,11]-seq[:,12],axis=1); mask=sw>1e-6
    avg=np.mean(sw[mask]) if np.any(mask) else 1.0; seq/=(avg+1e-6); return seq

def _smooth(arr,w=7): return np.convolve(arr,np.ones(w)/w,'same')

def detect_swing_phases(lm3d,fps,stroke_type='forehand'):
    T=len(lm3d); clamp=lambda x:int(np.clip(round(x),0,T-1))
    wr=lm3d[:,16,:]; wl=lm3d[:,15,:]
    shd=lm3d[:,11,0]-lm3d[:,12,0]; shc=_smooth(np.abs(np.diff(shd,prepend=shd[0])))
    vr=_smooth(np.concatenate([[0.],np.linalg.norm(np.diff(wr,axis=0),axis=1)])*fps)
    vl=_smooth(np.concatenate([[0.],np.linalg.norm(np.diff(wl,axis=0),axis=1)])*fps)
    p={}
    if stroke_type=='forehand':
        p['split_step']=clamp(T*0.03); p['unit_turn']=clamp(np.argmax(shc[:int(T*0.45)]))
        p['racket_drop']=clamp(np.argmax(lm3d[:int(T*0.65),16,1]))
        s,e=p['racket_drop'],clamp(T*0.92)
        p['contact']=clamp(s+np.argmax(vr[s:e])) if s<e else clamp(T*0.70)
        p['acceleration']=clamp((p['racket_drop']+p['contact'])/2)
        p['follow_through']=clamp(p['contact']+max(1,int(T*0.14)))
    elif stroke_type=='backhand':
        p['split_step']=clamp(T*0.03); p['unit_turn']=clamp(np.argmax(shc[:int(T*0.45)]))
        p['takeback']=clamp(np.argmin(lm3d[:int(T*0.60),15,0]))
        rl=int(np.argmax(lm3d[:int(T*0.70),15,1])); rr=int(np.argmax(lm3d[:int(T*0.70),16,1]))
        p['racket_drop']=clamp(rl if lm3d[rl,15,1]>lm3d[rr,16,1] else rr)
        s,e=p['racket_drop'],clamp(T*0.92)
        p['contact']=clamp(s+np.argmax(np.maximum(vl,vr)[s:e])) if s<e else clamp(T*0.70)
        p['follow_through']=clamp(p['contact']+max(1,int(T*0.14)))
    elif stroke_type=='serve':
        p['stance']=clamp(T*0.04); p['trophy']=clamp(np.argmin(lm3d[:int(T*0.55),16,1]))
        s=p['trophy']; e=clamp(T*0.80)
        p['racket_drop']=clamp(s+int(np.argmax(lm3d[s:e,16,1]))) if s<e else clamp(T*0.60)
        s2,e2=p['racket_drop'],clamp(T*0.95)
        p['contact']=clamp(s2+np.argmax(vr[s2:e2])) if s2<e2 else clamp(T*0.80)
        p['acceleration']=clamp((p['racket_drop']+p['contact'])/2)
        p['follow_through']=clamp(p['contact']+max(1,int(T*0.12)))
    return p

def build_phase_frame_map(user_phases,pro_phases,user_len,pro_len):
    common=sorted([k for k in user_phases if k in pro_phases],key=lambda k:user_phases[k])
    if len(common)<2:
        return np.clip(np.round(np.linspace(0,pro_len-1,user_len)).astype(int),0,pro_len-1)
    u_anc=[0]+[int(user_phases[k]) for k in common]+[user_len-1]
    p_anc=[0]+[int(pro_phases[k]) for k in common]+[pro_len-1]
    u_cl,p_cl=[u_anc[0]],[p_anc[0]]
    for u,p in zip(u_anc[1:],p_anc[1:]):
        if u>u_cl[-1]: u_cl.append(u); p_cl.append(p)
    if len(u_cl)<2:
        return np.clip(np.round(np.linspace(0,pro_len-1,user_len)).astype(int),0,pro_len-1)
    return np.clip(np.round(np.interp(np.arange(user_len),u_cl,p_cl)).astype(int),0,pro_len-1)

def _stroke_display_dur(phases,stroke_type,fps):
    fk={'forehand':'unit_turn','backhand':'unit_turn','serve':'trophy'}.get(stroke_type,'unit_turn')
    return max((phases.get('follow_through',0)-phases.get(fk,0))/fps,0.01)

def detect_video_speed_factor(lm3d, fps, stroke_type, manual=None):
    """
    Estimate slow-motion factor.  Returns sf where real_speed = display_speed / sf.
    sf=1.0 means real-time.  sf=0.1 means 10x slow-mo.

    Robustness fix: use only the racket_drop→follow_through window (the actual swing)
    rather than the full video duration.  This avoids false slow-mo detection when
    the video contains extra frames before/after the stroke.
    """
    if manual is not None: return float(manual)
    phases = detect_swing_phases(lm3d, fps, stroke_type)
    # Use the tightest meaningful window: racket_drop to follow_through
    start_key = {'forehand':'racket_drop','backhand':'racket_drop','serve':'racket_drop'}.get(stroke_type,'racket_drop')
    end_key   = 'follow_through'
    start_f = phases.get(start_key, 0)
    end_f   = phases.get(end_key, len(lm3d)-1)
    swing_frames = max(end_f - start_f, 1)
    # Expected swing window: forehand/backhand ~0.35s, serve ~0.40s
    expected_s = {'forehand':0.35,'backhand':0.38,'serve':0.40}.get(stroke_type, 0.35)
    display_s  = swing_frames / max(fps, 1.0)
    sf = expected_s / display_s
    # Clamp: allow up to 40x slow-mo (sf=0.025), never above 1.0 (real-time)
    return float(np.clip(sf, 0.025, 1.0))

def phases_to_seconds(phases,fps,sf):
    return {k:v/fps*sf for k,v in phases.items() if isinstance(v,(int,float,np.integer))}

def phase_resample_angles(angles_dict, phases, n_frames, n_out=101):
    """
    Resample each joint angle array to n_out points (0..100% stroke progress)
    using stroke phase frame indices as interpolation anchors.

    This ensures that when comparing user vs pro, each percentage point
    corresponds to the same stroke phase (unit turn, contact, etc.) for both,
    regardless of how many frames each video contains.

    phases:   dict of phase_name → frame_index
    n_frames: total frames in the video (len of the landmarks array)
    """
    # Build anchor list: (frame_idx, pct) from known phase names
    PHASE_PCTS = {
        'split_step': 0, 'unit_turn': 12, 'takeback': 18, 'stance': 0,
        'racket_drop': 35, 'trophy': 28,
        'acceleration': 55, 'contact': 72, 'follow_through': 90,
    }
    anchors = [(0, 0), (n_frames - 1, 100)]
    for name, pct in PHASE_PCTS.items():
        if name in phases:
            f = int(np.clip(phases[name], 0, n_frames - 1))
            anchors.append((f, pct))

    # Sort by pct, deduplicate, ensure strictly monotone in both dims
    anchors = sorted(set(anchors), key=lambda x: x[1])
    clean_f, clean_p = [anchors[0][0]], [anchors[0][1]]
    for f, p in anchors[1:]:
        if f > clean_f[-1] and p > clean_p[-1]:
            clean_f.append(f); clean_p.append(p)

    pct_axis   = np.linspace(0, 100, n_out)
    frame_axis = np.interp(pct_axis, clean_p, clean_f)
    frame_axis = np.clip(frame_axis, 0, n_frames - 1)

    resampled = {}
    for joint, arr in angles_dict.items():
        arr = np.asarray(arr, dtype=np.float64)
        src_idx = np.linspace(0, len(arr) - 1, len(arr))
        resampled[joint] = np.interp(frame_axis, src_idx, arr)
    return resampled

def resample_landmarks_linear(lm3d, n_out=101, max_frames=500):
    """Pure linear resample — identical to VideoOverlayWidget's frame mapping."""
    T = min(len(lm3d), max_frames)
    if T <= 0:
        return np.zeros((n_out, lm3d.shape[1], lm3d.shape[2]), dtype=np.float32)
    src = np.arange(T, dtype=np.float64)
    dst = np.linspace(0, T - 1, n_out)
    out = np.zeros((n_out, lm3d.shape[1], lm3d.shape[2]), dtype=np.float32)
    data = lm3d[:T]
    for lm in range(data.shape[1]):
        for d in range(data.shape[2]):
            out[:, lm, d] = np.interp(dst, src, data[:, lm, d].astype(np.float64))
    return out

def trim_resample_linear(lm3d, phases, stroke_type, n_out=101):
    """
    Resample landmark sequence to n_out frames using the IDENTICAL phase-anchor
    table as phase_resample_angles (PHASE_PCTS).

    This is the key insight: the joint-analysis graphs already look correct
    because phase_resample_angles maps every detected phase event to a fixed
    canonical percentage. Applying the exact same mapping to the 3D landmark
    arrays guarantees the skeleton display is frame-perfect with those graphs.

    At pct=72 both user and pro are at 'contact'.
    At pct=35 both are at 'racket_drop'.
    At pct=90 both are at 'follow_through'. Etc.
    Any detected phase that is present in both sequences is automatically
    synchronized — the more phases detected, the tighter the alignment.
    """
    T = len(lm3d)
    if T < 4:
        return resample_landmarks_linear(lm3d, n_out=n_out, max_frames=T)

    # Same table as phase_resample_angles
    PHASE_PCTS = {
        'split_step': 0, 'unit_turn': 12, 'takeback': 18, 'stance': 0,
        'racket_drop': 35, 'trophy': 28,
        'acceleration': 55, 'contact': 72, 'follow_through': 90,
    }

    # Build anchors: (frame_index, pct) from detected phases + sequence boundaries
    anchors = [(0, 0), (T - 1, 100)]
    for name, pct in PHASE_PCTS.items():
        if name in phases:
            f = int(np.clip(phases[name], 0, T - 1))
            anchors.append((f, pct))

    # Sort by pct; remove duplicates; enforce strict monotonicity in both dims
    anchors = sorted(set(anchors), key=lambda a: a[1])
    cf_list, cp_list = [anchors[0][0]], [anchors[0][1]]
    for f, p in anchors[1:]:
        if f > cf_list[-1] and p > cp_list[-1]:
            cf_list.append(f); cp_list.append(p)

    # Map output pct axis → source frame axis (piecewise linear)
    pct_axis   = np.linspace(0, 100, n_out)
    frame_axis = np.clip(np.interp(pct_axis, cp_list, cf_list), 0, T - 1)
    src_idx    = np.arange(T, dtype=np.float64)

    out = np.zeros((n_out, lm3d.shape[1], lm3d.shape[2]), dtype=np.float32)
    for lm in range(lm3d.shape[1]):
        for d in range(lm3d.shape[2]):
            out[:, lm, d] = np.interp(frame_axis, src_idx,
                                       lm3d[:, lm, d].astype(np.float64))
    return out



def compute_errors_from_arrays(ua_aligned, pa_aligned):
    """Compute per-joint mean error between two phase-aligned angle dicts."""
    errors = {}
    for joint in ua_aligned:
        if joint not in pa_aligned:
            continue
        u = ua_aligned[joint]
        p = pa_aligned[joint]
        diff = np.abs(u - p)
        errors[joint] = {
            'user_arr':   u,
            'pro_arr':    p,
            'mean_error': float(np.mean(diff)),
            'user_mean':  float(np.mean(u)),
            'pro_mean':   float(np.mean(p)),
        }
    return errors

def compute_racket_speed(lm3d, fps, sf):
    """
    Estimate racket head speed in shoulder-widths/second (real-world corrected).

    Key insight: dividing by sf is unreliable when the video contains frames
    outside the actual stroke (walking up, standing still) because sf → 0.025
    and division explodes.  Instead we:
    1. Compute raw frame-to-frame displacement * fps  (display units)
    2. Apply sf correction ONLY within the detected active swing window
       (top-10% of displacement frames), not globally
    3. Hard-clamp to physical max (114 sw/s ≈ 50 m/s ≈ 180 km/h racket)
    4. Typical recreational forehand: 30-60 sw/s → 13-26 m/s → 48-94 km/h ball
    """
    head = lm3d[:,16,:] + (lm3d[:,16,:] - lm3d[:,14,:]) * 1.8
    raw  = np.concatenate([[0.], np.linalg.norm(np.diff(head, axis=0), axis=1)]) * fps
    disp = _smooth(raw, w=9)

    # Active swing = frames above 20th percentile of non-zero speeds
    nonzero = disp[disp > 0.01]
    if len(nonzero) < 5:
        return {'racket_speed_display': disp, 'racket_speed_real': disp,
                'peak_speed_real': float(np.max(disp))}

    # Apply sf correction only where motion is actually happening
    # Use a soft correction: real = disp / sf, but cap sf influence
    sf_safe = float(np.clip(sf, 0.10, 1.0))   # never divide by less than 0.10
    real = _smooth(disp / sf_safe, w=5)

    # Hard physical cap: 114 sw/s ≈ 50 m/s racket head (world-class serve)
    real = np.clip(real, 0.0, 114.0)

    return {'racket_speed_display': disp, 'racket_speed_real': real,
            'peak_speed_real': float(np.max(real))}

def estimate_ball_speeds(rs_cont_real, stroke_type='forehand'):
    """
    Estimate ball exit speed in km/h, calibrated to real-world benchmarks.

    Benchmark targets (mph → km/h):
      Recreational/beginner groundstroke:  40–60 mph  =  64– 97 km/h
      High-school varsity groundstroke:    60–75 mph  =  97–121 km/h  (top: 80 mph = 129)
      College/club groundstroke:           70–90 mph  = 113–145 km/h
      Pro rally groundstroke:              75–90 mph  = 121–145 km/h
      Pro offensive groundstroke:         100+ mph   = 161+ km/h
      HS varsity 1st serve:               70–90 mph  = 113–145 km/h
      Pro 1st serve:                     110–130 mph  = 177–209 km/h

    Rather than trusting the raw sw/s value (which varies wildly with slow-mo
    detection accuracy), we use it only as a RELATIVE indicator: we map the
    percentile of the contact speed within its physical range to a target km/h
    range for the stroke type, so the output is always plausible.
    """
    if not rs_cont_real or rs_cont_real <= 0: return None, None

    # Raw racket m/s — the sw/s value × 0.44m shoulder width
    racket_ms_raw = float(rs_cont_real * SW_METERS)

    # Physical racket speed range in m/s:
    #   ~5 m/s  = 18 km/h  (gentle beginner push)
    #   ~50 m/s = 180 km/h (world-record serve racket head)
    # Most club players: 10–30 m/s
    racket_ms = float(np.clip(racket_ms_raw, 3.0, 55.0))

    # Map racket speed to ball exit speed using stroke-specific calibrated ranges.
    # Low end = beginner racket speed (~8 m/s), high end = elite (~50 m/s)
    # Output km/h spans the full benchmark range.
    R_LO, R_HI = 8.0, 50.0   # racket speed bounds for interpolation

    target_ranges = {
        #                   (lo_km/h, hi_km/h)
        'forehand':         (60.0,  160.0),   # beginner 60, Alcaraz 160
        'backhand':         (55.0,  145.0),
        'serve':            (90.0,  210.0),   # beginner serve 90, Isner 210
    }
    lo_ball, hi_ball = target_ranges.get(stroke_type, (55.0, 155.0))

    # Linear interpolation within the target range
    t = float(np.clip((racket_ms - R_LO) / (R_HI - R_LO), 0.0, 1.0))
    ball_kmh = lo_ball + t * (hi_ball - lo_ball)

    # Final sanity clamp — these are absolute outer limits
    ball_kmh = float(np.clip(ball_kmh, lo_ball, hi_ball))
    return None, ball_kmh

def classify_contact_point(lm3d,cf):
    if cf>=len(lm3d): return "Unknown",0.0
    lm=lm3d[cf]; hip_x=float((lm[23,0]+lm[24,0])/2)
    hx=float(lm[16,0]+(lm[16,0]-lm[14,0])*1.8); diff=hx-hip_x
    if diff>0.70: label="Well In Front (Power Zone)"
    elif diff>0.30: label="In Front (Good)"
    elif diff>0.00: label="Slightly In Front"
    elif diff>-0.25: label="To the Side"
    else: label="Behind Body (Late)"
    return label,diff

def estimate_racket_face_angle(lm3d,cf):
    if cf>=len(lm3d): return 0.
    lm=lm3d[cf]
    return float(np.degrees(np.arctan2(lm[16,0]-lm[14,0],-(lm[16,1]-lm[14,1]))))

def estimate_topspin_from_swing(lm3d, phases, stroke_type='forehand'):
    """
    Estimate topspin RPM.
    Serves: hard 400 RPM (flat serve default). The structural upward arc of
    the trophy→contact swing is NOT a topspin brush. Serve spin comes from
    forearm pronation which is invisible in wrist-position data.
    Groundstrokes: brush angle from last 25% of racket_drop→contact window,
    calibrated to ATP reference (forehand avg 2700-3000 RPM).
    """
    T = len(lm3d)
    if stroke_type == 'serve':
        return 400.0   # flat 1st serve; pronation-based spin not measurable here
    if T < 4: return 1800.0
    head = lm3d[:,16,:] + (lm3d[:,16,:] - lm3d[:,14,:]) * 1.8
    f_drop    = int(np.clip(phases.get('racket_drop', int(T * 0.35)), 0, T-1))
    f_contact = int(np.clip(phases.get('contact',     int(T * 0.70)), 0, T-1))
    if f_contact <= f_drop: return 1800.0
    brush_start = max(f_drop + 1, int(f_drop + 0.75 * (f_contact - f_drop)))
    delta_y = float(head[brush_start, 1]) - float(head[f_contact, 1])  # +ve = upward
    delta_x = float(abs(head[f_contact, 0] - head[brush_start, 0])) + 1e-6
    brush_angle = float(np.degrees(np.arctan2(max(0.0, delta_y), delta_x)))
    base_rpm = float(np.interp(brush_angle, [0,15,35,55,70], [600,1500,2800,3800,5000]))
    if stroke_type == 'backhand':
        return float(np.clip(base_rpm * 0.80, 800, 3200))
    return float(np.clip(base_rpm, 1000, 5000))

def compute_ball_trajectory_physics(v0_racket_ms,racket_angle_deg,
                                     stroke_type='forehand',
                                     contact_height_m=0.80,wind_ms=0.0,
                                     override_spin_rpm=None,
                                     override_launch_deg=None,
                                     override_exit_kmh=None):
    g=9.81;rho=1.225;m=0.057;r=0.033;A=np.pi*r**2;Cd=0.55;D=0.5*rho*A*Cd/m
    if stroke_type=='serve': contact_height_m=max(contact_height_m,2.4)

    # ── Ball exit speed ───────────────────────────────────────────────
    if override_exit_kmh is not None and override_exit_kmh > 0:
        v0 = float(np.clip(override_exit_kmh / 3.6, 8.0, 65.0))
    else:
        racket_ms = float(np.clip(v0_racket_ms, 3.0, 55.0))
        R_LO, R_HI = 8.0, 50.0
        target_kmh = {'forehand':(65.0,175.0),'backhand':(60.0,160.0),'serve':(95.0,225.0)}.get(stroke_type,(65.0,165.0))
        t_interp = float(np.clip((racket_ms - R_LO)/(R_HI - R_LO), 0.0, 1.0))
        v0 = float(np.clip((target_kmh[0] + t_interp*(target_kmh[1]-target_kmh[0]))/3.6, 8.0, 65.0))

    # ── Topspin RPM ───────────────────────────────────────────────────
    if override_spin_rpm is not None:
        spin_RPM = float(np.clip(override_spin_rpm, 0.0, 5000.0))
    else:
        base_rpm = {'forehand':2800.0,'backhand':2000.0,'serve':400.0}.get(stroke_type,2000.0)
        rpm_adj  = -racket_angle_deg * 40.0
        if stroke_type == 'serve':
            spin_RPM = float(np.clip(base_rpm, 200.0, 2500.0))
        else:
            spin_RPM = float(np.clip(base_rpm + rpm_adj, 600.0, 5000.0))
    # Serve: suppress Magnus for flat serves (<800 RPM)
    if stroke_type == 'serve':
        magnus_blend = float(np.clip((spin_RPM - 400.0) / 800.0, 0.0, 1.0))
        omega = spin_RPM * magnus_blend * 2.0 * np.pi / 60.0
    else:
        omega = spin_RPM * 2.0 * np.pi / 60.0

    # ── Launch angle ──────────────────────────────────────────────────
    if override_launch_deg is not None:
        launch_deg = float(override_launch_deg)
    else:
        if stroke_type == 'serve':
            launch_deg = -1.5
        else:
            base_ang = {'forehand':13.0,'backhand':12.0}.get(stroke_type,13.0)
            face_corr = -racket_angle_deg * 0.06
            launch_deg = float(np.clip(base_ang+face_corr, 7.0, 25.0))
    launch_rad = np.radians(launch_deg)
    vx_=v0*np.cos(launch_rad); vy_=v0*np.sin(launch_rad); vz_=0.0
    def magnus_down(vm,om):
        if vm<0.5 or om<1.0: return 0.0
        return 0.5*rho*A*float(np.clip(2.5*om*r/vm,0.0,0.55))*vm**2/m
    dt=0.002; x=[-1.0]; y=[contact_height_m]; z=[0.0]
    n_bounces=0; bounce_x=None; bounce_z=0.0; t=0.0
    bounce_xs=[]
    while t<8.0 and n_bounces<3:
        vm=np.sqrt(vx_**2+vy_**2+vz_**2)
        if vm<0.3: break
        vx_rel=vx_-wind_ms; vm_rel=np.sqrt(vx_rel**2+vy_**2+vz_**2)
        Fm=magnus_down(vm,omega); ax_=-D*vx_rel*vm_rel; ay_=-g-D*vy_*vm_rel-Fm
        nx=x[-1]+vx_*dt; ny=y[-1]+vy_*dt
        vx_+=ax_*dt; vy_+=ay_*dt
        if ny<=0.0:
            ny=0.0; n_bounces+=1
            if n_bounces==1: bounce_x=nx; bounce_z=0.0
            bounce_xs.append(float(nx))
            vy_=+0.72*abs(vy_)*(0.90**(n_bounces-1))   # bounce UP, energy loss each bounce
            vx_+=omega*r*0.72*0.22*0.6; omega*=0.45
            if n_bounces>=3:
                x.append(nx); y.append(0.0); z.append(0.0); break
        x.append(nx); y.append(max(ny,0.0)); z.append(0.0)
        t+=dt
        if nx>COURT_LEN+6: break
    xa=np.array(x); ya=np.array(y); za=np.array(z)
    if bounce_x is None:
        t_est=(vy_+np.sqrt(max(0,vy_**2+2*g*contact_height_m)))/g
        bounce_x=float(vx_*t_est+x[-1])
    pre=xa<=NET_X
    if pre.any() and (~pre).any():
        xb_=xa[pre][-1];xa__=xa[~pre][0];yb_=ya[pre][-1];ya__=ya[~pre][0]
        y_net=float(yb_+(NET_X-xb_)/(xa__-xb_+1e-8)*(ya__-yb_))
    else: y_net=float(contact_height_m)
    cm=(xa>=-2.0)&(xa<=COURT_LEN+6)
    if cm.any(): xa=xa[cm]; ya=ya[cm]; za=za[cm]
    over_net=bool(y_net>NET_H_CTR)
    if stroke_type=='serve': in_c=(NET_X<bounce_x<=SVC_LINE_O) and (abs(bounce_z)<=SGL_W/2)
    else: in_c=(NET_X<bounce_x<=COURT_LEN) and (abs(bounce_z)<=SGL_W/2)
    return dict(x=xa,y=ya,z=za,x_land=float(bounce_x),z_land=float(bounce_z),
                y_net=float(y_net),over_net=over_net,in_court=in_c,
                bounce_xs=bounce_xs,launch_deg=launch_deg,v0_ms=v0,
                spin_RPM=spin_RPM,wind_ms=float(wind_ms),stroke_type=stroke_type)

def generate_detailed_coaching(res, stroke_type, pro_name="Pro"):
    """Rule-based coaching report. Returns clean HTML for QTextEdit."""
    errors=res.get('errors',{}); pu_s=res.get('phases_user_s',{})
    pp_s=res.get('phases_pro_s',{}); contact=res.get('contact_label','')
    rack_ang=res.get('racket_angle',0.0); peak_rs=res.get('peak_speed_real',0.0)
    rs_cont=res.get('rs_at_contact',0.0); sf_u=res.get('speed_factor_user',1.0)
    _,out_k=estimate_ball_speeds(rs_cont,stroke_type)
    pdefs=STROKE_PHASES.get(stroke_type,STROKE_PHASES['forehand'])
    pkeys=[p[0] for p in pdefs]; pnames=[p[1] for p in pdefs]

    # ── Clean, readable colour palette ───────────────────────────────
    BG='#131929'; TXT='#e8eaf0'; SUB='#8a9bb5'; DIM='#5a6a80'
    GREEN_C='#3ddc84'; AMBER='#ffb74d'; RED_C='#ef5350'; BLUE_C='#64b5f6'
    CARD='#1e2d45'; LINE='#2a3f5a'

    def h1(t):
        return (f'<p style="font-size:19px;font-weight:700;color:{BLUE_C};'
                f'margin:0 0 12px 0;padding-bottom:10px;'
                f'border-bottom:2px solid {LINE};">{t}</p>')
    def h2(t, color=BLUE_C):
        # Section header with left accent bar — no background fill
        return (f'<p style="font-size:14px;font-weight:700;color:{color};'
                f'margin:20px 0 10px 0;padding:6px 0 6px 12px;'
                f'border-left:3px solid {color};">{t}</p>')
    def stat_row(label, value, color=GREEN_C):
        return (f'<span style="color:{SUB};font-size:13px;">{label}: </span>'
                f'<span style="color:{color};font-size:13px;font-weight:600;">{value}</span>'
                f'&nbsp;&nbsp;&nbsp;')
    def finding(icon, severity, title, body, drill_text=None):
        c = {' ✅':GREEN_C,' 🟡':AMBER,' 🔴':RED_C}.get(' '+icon, TXT)
        # Plain layout — icon + bold title + normal body. No card/box.
        s = (f'<div style="margin:0 0 14px 0;">'
             f'<p style="margin:0 0 3px 0;">'
             f'<span style="font-size:14px;">{icon}</span>'
             f'<span style="color:{c};font-size:13px;font-weight:700;'
             f'margin-left:6px;">{title}</span>'
             f'<span style="color:{DIM};font-size:11px;margin-left:8px;">{severity}</span>'
             f'</p>'
             f'<p style="color:#c8d0dc;font-size:13px;line-height:1.7;'
             f'margin:0 0 0 24px;">{body}</p>')
        if drill_text:
            s += (f'<p style="color:{GREEN_C};font-size:12px;font-style:italic;'
                  f'border-left:2px solid {GREEN_C};padding:4px 8px;'
                  f'margin:5px 0 0 24px;line-height:1.5;">'
                  f'🏃 Drill: {drill_text}</p>')
        s += '</div>'
        return s
    def divider():
        return f'<hr style="border:none;border-top:1px solid {LINE};margin:16px 0;">'

    out=[f'<div style="font-family:Segoe UI,Helvetica,Arial,sans-serif;'
         f'font-size:13px;background:{BG};color:{TXT};padding:16px 20px;">']

    # Header & speeds
    spd=float(np.clip(rs_cont*SW_METERS,0,55)); pk=float(np.clip(peak_rs*SW_METERS,0,55))
    out.append(h1(f'{stroke_type.capitalize()} Report vs {pro_name}'))
    out.append(f'<p style="margin:10px 0 4px 0;">')
    out.append(stat_row('Racket at contact', f'{spd*3.6:.0f} km/h'))
    out.append(stat_row('Peak racket', f'{pk*3.6:.0f} km/h'))
    if out_k: out.append(stat_row('Est. ball exit', f'~{out_k:.0f} km/h'))
    if sf_u<0.85: out.append(f'<br><span style="color:{AMBER};font-size:12px;">'
                              f'(slow-motion {sf_u:.3f}× correction applied)</span>')
    out.append('</p>')
    out.append(divider())

    # ── Contact Point ─────────────────────────────────────────────────
    out.append(h2('1. Contact Point'))
    if 'Well In Front' in contact or 'Power Zone' in contact:
        out.append(finding('✅','Excellent',contact,
            'Hitting well in front of your body with a full arm extension. '
            'This is the power zone where racket head speed transfers most efficiently into the ball. '
            'Keep reinforcing this habit.'))
    elif 'In Front' in contact:
        out.append(finding('✅','Good',contact,
            'Meeting the ball in front of your body — solid position. '
            'Try to extend just a little further forward to maximise leverage and add pace.'))
    elif 'Slightly In Front' in contact:
        out.append(finding('🟡','Improvable',contact,
            'You are getting the ball slightly in front, but not in the full power zone. '
            'Earlier unit turn will give your arm more time to extend forward.',
            'Feed drill: toss balls from a basket and step into each one, '
            'consciously making contact over your front foot.'))
    elif 'Side' in contact:
        out.append(finding('🔴','Priority fix','Contact to the side',
            'The ball is reaching you beside your hip — the racket has already passed peak speed. '
            'You are hitting with a decelerating racket, costing significant power and consistency.',
            'Shadow swing: draw a chalk line 50 cm ahead of your front foot. '
            'Every shadow contact must happen over that line.'))
    elif 'Behind' in contact:
        out.append(finding('🔴','Highest priority','Late contact — behind the body',
            'Contact behind your body forces you to muscle the shot with just your arm, '
            'losing all hip and shoulder rotation power. '
            'Root cause: unit turn is starting too late.',
            'Read & react: the instant your opponent starts their backswing, '
            'begin your shoulder turn. Do not wait to track the ball.'))
    out.append(divider())

    # ── Phase Timing ─────────────────────────────────────────────────
    out.append(h2('2. Phase Timing'))
    any_phase=False
    goods=[]
    for i in range(len(pkeys)-1):
        k0,k1=pkeys[i],pkeys[i+1]
        if not all(k in pu_s for k in(k0,k1)) or not all(k in pp_s for k in(k0,k1)): continue
        u_dt=pu_s[k1]-pu_s[k0]; p_dt=pp_s[k1]-pp_s[k0]
        if p_dt<0.01: continue
        rel=(u_dt-p_dt)/p_dt; name=pnames[i+1]; diff_ms=(u_dt-p_dt)*1000
        if rel>0.35:
            any_phase=True
            out.append(finding('🔴','Slow',name,
                f'Your {name.lower()} takes {u_dt:.2f}s vs {pro_name} ({p_dt:.2f}s) '
                f'(+{diff_ms:.0f}ms). Be more explosive and decisive through this phase.'))
        elif rel>0.15:
            any_phase=True
            out.append(finding('🟡','Slightly slow',name,
                f'{u_dt:.2f}s vs {pro_name} (+{diff_ms:.0f}ms, pro: {p_dt:.2f}s). '
                f'A small improvement here will noticeably add pace.'))
        elif rel<-0.35:
            any_phase=True
            out.append(finding('🟡','Rushing',name,
                f'{u_dt:.2f}s vs {pro_name} ({diff_ms:.0f}ms, pro: {p_dt:.2f}s). '
                f'Ensure you are completing this phase fully before accelerating.'))
        else:
            goods.append(name)
    if goods:
        out.append(f'<p style="color:{GREEN_C};font-size:13px;margin:6px 0;">'
                   f'✅ Good timing: {", ".join(goods)}</p>')
    if not any_phase and not goods:
        out.append(f'<p style="color:{SUB};font-size:13px;margin:6px 0;">'
                   f'Phase timing data not available.</p>')
    out.append(divider())

    # ── Joint Mechanics ───────────────────────────────────────────────
    out.append(h2('3. Joint Mechanics'))
    out.append(f'<p style="color:{SUB};font-size:12px;margin:0 0 8px 0;">'
               f'Sorted by deviation from {pro_name}. '
               f'Each degree of consistent deviation compounds into visible technique differences.</p>')

    COACHING={
        'right_elbow':{
            'less':('Hitting elbow too bent at contact',
                    'A bent elbow shortens your lever arm — you lose reach and whipping acceleration. '
                    'Aim to straighten to within 20° of fully extended as you approach contact.',
                    'Wall stop drill: shadow swing and freeze at contact. Check elbow is near-straight.'),
            'more':('Hitting elbow too straight through swing',
                    'A locked elbow limits your ability to brush the ball for topspin. '
                    'Keep a natural soft bend through the acceleration phase.',None)},
        'right_shoulder':{
            'less':('Insufficient shoulder rotation',
                    'Shoulder rotation is the engine of the groundstroke. '
                    'Drive your shoulder toward the opposite net post through and past contact. '
                    'More rotation = longer acceleration arc = more speed.',
                    'Eyes-closed drill: focus entirely on feeling the shoulder drive through contact.'),
            'more':('Over-rotating shoulder past contact',
                    'Spinning past contact costs accuracy — the racket face moves off-target. '
                    'Think of stabilising your shoulder AT contact for one beat before following through.',None)},
        'hip_rotation':{
            'less':('Hips not contributing to kinetic chain',
                    'Hips should fire at least 50ms before shoulders. '
                    'If you skip hip drive the arm has to compensate, '
                    'creating inconsistency and capping your power ceiling.',
                    'Band drill: resistance band around waist. Shadow swing: hips → shoulders → arm with a clear pause between.'),
            'more':('Hips spinning out excessively',
                    'Over-rotating hips shifts weight away from the ball. '
                    'Focus on hip stability at the contact moment to transfer force upward.',None)},
        'right_knee':{
            'less':('Hitting-side knee too straight',
                    'Knee bend loads your legs like a coil — you release into the ball with an upward push. '
                    'Without this leg drive you are hitting entirely with upper body.',
                    'Low ball drill: hit feeds dropped below knee height, forcing deep bend and upward drive.'),
            'more':('Excessive knee bend on hitting side',
                    'Aim for 20–30° knee bend at ready position and extend through contact.',None)},
        'left_elbow':{
            'less':('Non-dominant arm too passive',
                    'The non-dominant arm acts as a counterweight and trigger. '
                    'Pulling it sharply back during the swing slingshots the hitting arm forward.',
                    'Towel drill: hold a towel in non-dominant hand, pull it sharply back as you swing.'),
            'more':('Non-dominant elbow too bent',
                    'A bent front elbow restricts shoulder rotation. '
                    'Keep it more extended during the backswing coil.',None)},
        'left_shoulder':{
            'less':('Front shoulder not coiling enough',
                    'More shoulder coil during unit turn = more elastic energy stored. '
                    'Turn until your back faces the net, then uncoil through the ball.',
                    'Exaggeration drill: during shadow swings rotate until back fully faces net, hold for one beat.'),
            'more':('Front shoulder opening too early',
                    'Early shoulder opening is the most common power leak. '
                    'Keep the front shoulder closed until the racket is almost at contact.',
                    'Fence drill: stand a foot from fence, front shoulder must point at fence throughout backswing.')},
        'left_knee':{
            'less':('Support knee too straight',
                    'A bent support knee lowers your centre of gravity and improves balance, '
                    'allowing your hitting side to drive upward more explosively.',
                    'Athletic base: spend 30s before every session in tennis ready position — both knees bent.'),
            'more':('Support leg lunge too wide',
                    'Too wide a stance destabilises your base and slows recovery. '
                    'Keep a controlled athletic bend rather than a wide lunge.',None)},
    }
    shown=0
    for joint,je in sorted(errors.items(),key=lambda x:-x[1]['mean_error']):
        if je['mean_error']<8.0 or shown>=4: break
        u_m=je.get('user_mean',0); p_m=je.get('pro_mean',0); err=je['mean_error']
        direction='less' if u_m<p_m else 'more'
        coaching_t=COACHING.get(joint,{}).get(direction,None)
        jdesc=JOINT_DESCRIPTIONS.get(joint,(joint.replace('_',' ').title(),''))
        icon='🔴' if err>20 else('🟡' if err>12 else '🟢')
        sev=f'{err:.1f}° avg deviation'
        if coaching_t:
            out.append(finding(icon, sev, coaching_t[0], coaching_t[1], coaching_t[2]))
        else:
            out.append(finding(icon, sev, jdesc[0], 'Deviation noted — compare with Joint Analysis graphs.'))
        shown+=1
    if shown==0:
        out.append(f'<p style="color:{GREEN_C};font-size:13px;margin:6px 0;">'
                   f'✅ Joint mechanics closely match {pro_name}.</p>')
    out.append(divider())

    # ── Racket Face ───────────────────────────────────────────────────
    if stroke_type in ('forehand','backhand'):
        out.append(h2('4. Racket Face'))
        if rack_ang<-20:
            out.append(finding('🟡',f'{abs(rack_ang):.0f}° tilted back','Heavy topspin, depth risk',
                'A tilted-back face creates heavy topspin — powerful, but risks hitting long. '
                'If shots are going deep, close the face 5–10° or brush up more steeply.'))
        elif rack_ang>20:
            out.append(finding('🟡',f'{rack_ang:.0f}° open face','Net risk',
                'An open face directs the ball downward. '
                'Open the face slightly and brush up the back of the ball rather than driving through it.'))
        else:
            out.append(finding('✅',f'{rack_ang:.1f}° from vertical','Good',
                'Racket face angle at contact is solid — healthy balance between power and topspin.'))
        out.append(divider())

    # ── Summary ────────────────────────────────────────────────────────
    avg_err=float(np.mean([je['mean_error'] for je in errors.values()])) if errors else 0
    if avg_err<10:   rating,rc='Excellent',GREEN_C; note='Technique closely mirrors the pro. Focus on consistency under match pressure.'
    elif avg_err<18: rating,rc='Developing',AMBER;  note='Solid foundation. Target the red-flagged items first.'
    else:            rating,rc='Needs work',RED_C;  note='Several patterns differ from the pro. Prioritise the top 2 joint issues with dedicated drills.'
    out.append(divider())
    out.append(f'<p style="font-size:13px;margin:0 0 4px 0;">'
               f'<span style="color:{rc};font-weight:700;font-size:14px;">Overall: {rating}</span>'
               f'&nbsp;&nbsp;<span style="color:{DIM};font-size:12px;">avg deviation {avg_err:.1f}° from {pro_name}</span></p>'
               f'<p style="color:{SUB};font-size:13px;line-height:1.6;margin:0 0 8px 0;">{note}</p>')
    out.append('</div>')
    return ''.join(out)


def generate_feedback(res,stroke_type):
    """Short feedback list for the left panel."""
    lines=[]; pu_s=res.get('phases_user_s',{}); pp_s=res.get('phases_pro_s',{})
    contact=res.get('contact_label',''); peak_rs=res.get('peak_speed_real',0.0)
    rs_cont=res.get('rs_at_contact',0.0); _,out_k=estimate_ball_speeds(rs_cont,stroke_type)
    pdefs=STROKE_PHASES.get(stroke_type,STROKE_PHASES['forehand'])
    pkeys=[p[0] for p in pdefs]; pnames=[p[1] for p in pdefs]
    for i in range(len(pkeys)-1):
        k0,k1=pkeys[i],pkeys[i+1]
        if not all(k in pu_s for k in(k0,k1)) or not all(k in pp_s for k in(k0,k1)): continue
        u_dt=pu_s[k1]-pu_s[k0]; p_dt=pp_s[k1]-pp_s[k0]
        if p_dt<0.01: continue
        rel=(u_dt-p_dt)/p_dt
        if rel>0.20: lines.append(f"{pnames[i+1]}: {u_dt:.2f}s vs pro {p_dt:.2f}s — be more compact")
        elif rel<-0.20: lines.append(f"{pnames[i+1]}: don't rush — complete the preparation")
    if 'Behind' in contact: lines.append("LATE CONTACT — start unit turn earlier")
    elif 'To the Side' in contact: lines.append("Contact to the side — extend forward more")
    lines.append(f"Racket at contact: {rs_cont*SW_METERS*3.6:.0f} km/h  (peak {peak_rs*SW_METERS*3.6:.0f} km/h)")
    if out_k: lines.append(f"Est. ball exit: ~{out_k:.0f} km/h")
    if not lines: lines.append(f"Solid {stroke_type}! Technique matches the pro closely.")
    return lines


class AIFeedbackThread(QThread):
    finished=pyqtSignal(str); error_sig=pyqtSignal(str)
    def __init__(self,analysis_data): super().__init__(); self.data=analysis_data
    def run(self):
        try:
            import urllib.request
            d=self.data; st=d['stroke_type']
            pdefs=STROKE_PHASES.get(st,STROKE_PHASES['forehand'])
            pkeys=[p[0] for p in pdefs]; pnames=[p[1] for p in pdefs]
            pu_s=d.get('phases_user_s',{}); pp_s=d.get('phases_pro_s',{})
            phase_txt=[]
            for i in range(len(pkeys)-1):
                k0,k1=pkeys[i],pkeys[i+1]
                if not all(k in pu_s for k in(k0,k1)) or not all(k in pp_s for k in(k0,k1)): continue
                u_dt=pu_s[k1]-pu_s[k0]; p_dt=pp_s[k1]-pp_s[k0]
                if p_dt<0.01: continue
                pct=(u_dt-p_dt)/p_dt*100
                phase_txt.append(f"  {pnames[i+1]}: user={u_dt:.3f}s  pro={p_dt:.3f}s  ({pct:+.0f}%)")
            errors=d.get('errors',{})
            joint_txt=[f"  {jn}: error={je['mean_error']:.1f} deg (user={je.get('user_mean',0):.1f} pro={je.get('pro_mean',0):.1f})"
                       for jn,je in errors.items()]
            prompt=(
                f"You are an expert tennis coach. Analyse this player's {st} and give specific, actionable coaching feedback.\n\n"
                f"Player stats:\n"
                f"  Contact position: {d.get('contact_label','')}\n"
                f"  Racket face angle vs vertical: {d.get('racket_angle',0):.1f} deg\n"
                f"  Racket speed at contact: {d.get('rs_at_contact',0)*SW_METERS*3.6:.0f} km/h\n"
                f"  Peak racket speed: {d.get('peak_speed_real',0)*SW_METERS*3.6:.0f} km/h\n\n"
                f"Phase timing (user vs pro):\n"+'\n'.join(phase_txt)+"\n\n"
                f"Joint angle deviations from pro:\n"+'\n'.join(joint_txt)+"\n\n"
                f"Write 5-6 coaching bullet points. For each issue:\n"
                f"- Name the issue clearly\n"
                f"- Explain WHY it hurts the shot\n"
                f"- Give one specific drill or cue to fix it\n"
                f"Start each bullet with a sport emoji (e.g. 🎾 ⚡ 💪). Keep it under 400 words."
            )
            headers={"Content-Type":"application/json","anthropic-version":"2023-06-01"}
            api_key=os.environ.get("ANTHROPIC_API_KEY","")
            if api_key: headers["x-api-key"]=api_key
            payload=json.dumps({"model":"claude-sonnet-4-5","max_tokens":700,
                                  "messages":[{"role":"user","content":prompt}]}).encode('utf-8')
            req=urllib.request.Request("https://api.anthropic.com/v1/messages",
                data=payload,headers=headers,method='POST')
            with urllib.request.urlopen(req,timeout=30) as resp:
                result=json.loads(resp.read().decode('utf-8'))
            self.finished.emit(result['content'][0]['text'])
        except Exception as e:
            self.error_sig.emit(
                f"AI coaching unavailable.\n\nTo enable: set ANTHROPIC_API_KEY environment variable.\n\nError: {e}")


class ProcessingThread(QThread):
    progress=pyqtSignal(str,int); finished=pyqtSignal(dict); error=pyqtSignal(str)
    def __init__(self,vp,st,pd,lbl="",usr_sf=None,pro_sf=None):
        super().__init__(); self.vp=vp; self.st=st; self.pd=pd
        self.lbl=lbl; self.usr_sf=usr_sf; self.pro_sf=pro_sf
    def run(self):
        try:
            from processing.pose_extractor import PoseExtractor
            from processing.comparator import PoseComparator
            ext=PoseExtractor(); cmp=PoseComparator()
            self.progress.emit("Extracting pose...",10)
            user=ext.extract_from_video(self.vp)
            self.progress.emit("Joint angles...",30)
            ua=ext.compute_joint_angles(user['landmarks_3d'])
            pro_vis,med_vid=self._pick_best_pro(ua)
            self.progress.emit("Pro angles...",42)
            pa=ext.compute_joint_angles(pro_vis['landmarks_3d'])
            self.progress.emit("Phases & speed...",52)
            fps_u=float(user.get('fps',30)); fps_p=float(pro_vis.get('fps',30))
            sf_u=detect_video_speed_factor(user['landmarks_3d'],fps_u,self.st,self.usr_sf)
            sf_p=detect_video_speed_factor(pro_vis['landmarks_3d'],fps_p,self.st,self.pro_sf)
            pu=detect_swing_phases(user['landmarks_3d'],fps_u,self.st)
            pp=detect_swing_phases(pro_vis['landmarks_3d'],fps_p,self.st)
            pu_s=phases_to_seconds(pu,fps_u,sf_u); pp_s=phases_to_seconds(pp,fps_p,sf_p)
            rs_u=compute_racket_speed(user['landmarks_3d'],fps_u,sf_u)
            rs_p=compute_racket_speed(pro_vis['landmarks_3d'],fps_p,sf_p)
            cf=pu.get('contact',0)
            rs_cont=float(rs_u['racket_speed_real'][min(cf,len(rs_u['racket_speed_real'])-1)])
            cl,cd=classify_contact_point(user['landmarks_3d'],cf)
            ra=estimate_racket_face_angle(user['landmarks_3d'],cf)
            spin_rpm=estimate_topspin_from_swing(user['landmarks_3d'],pu,self.st)
            lm_cf=user['landmarks_3d'][min(cf,len(user['landmarks_3d'])-1)]
            contact_h=max(0.3,1.1-float(lm_cf[16,1])*SW_METERS)
            if self.st=='serve': contact_h=max(contact_h,2.2)
            self.progress.emit("Comparing...",70)
            # Phase-aligned joint comparison: resample both to 101 stroke-progress
            # points so each % point corresponds to the same phase, not the same
            # raw frame index.  This ensures same-video user/pro gives ~0 error.
            ua_aligned = phase_resample_angles(ua, pu, len(user['landmarks_3d']))
            pa_aligned = phase_resample_angles(pa, pp, len(pro_vis['landmarks_3d']))
            errors = compute_errors_from_arrays(ua_aligned, pa_aligned)
            self.progress.emit("Feedback...",84)
            fb=generate_feedback(dict(errors=errors,phases_user_s=pu_s,phases_pro_s=pp_s,
                contact_label=cl,racket_angle=ra,peak_speed_real=rs_u['peak_speed_real'],
                rs_at_contact=rs_cont),self.st)
            self.finished.emit(dict(
                user_data=user,pro_data=pro_vis,user_ang=ua,pro_ang=pa,
                errors=errors,feedback=fb,pro_label=self.lbl,medoid_video=med_vid,
                phases_user=pu,phases_pro=pp,phases_user_s=pu_s,phases_pro_s=pp_s,
                speed_factor_user=sf_u,speed_factor_pro=sf_p,
                racket_speed_real=rs_u['racket_speed_real'],
                pro_racket_speed_real=rs_p['racket_speed_real'],
                peak_speed_real=rs_u['peak_speed_real'],
                rs_at_contact=rs_cont,contact_label=cl,contact_diff=cd,
                racket_angle=ra,spin_rpm=spin_rpm,contact_height_m=contact_h,stroke_type=self.st))
        except Exception:
            import traceback; self.error.emit(traceback.format_exc())
    def _pick_best_pro(self,user_angles):
        pd=self.pd
        try:
            from data.reference_builder import find_best_match_from_ref
            best_idx=find_best_match_from_ref(user_angles,pd)
            if 'all_landmarks_3d' in pd and best_idx<len(pd['all_landmarks_3d']):
                tl=pd.get('target_len',120)
                pro_vis={'landmarks_3d':pd['all_landmarks_3d'][best_idx],
                         'landmarks_2d':pd['all_landmarks_2d'][best_idx],
                         'frame_indices':np.arange(tl,dtype=np.int32),
                         'fps':float(pd['all_fps'][best_idx]),'total_frames':tl}
                sv=pd.get('source_videos',[])
                med=sv[best_idx] if best_idx<len(sv) else pd.get('medoid_video','')
                return pro_vis,med
        except Exception: pass
        pro_vis=pd['medoid_data'] if 'medoid_data' in pd else pd
        med=pd.get('medoid_video') if 'medoid_data' in pd else None
        return pro_vis,med

class PanelTooltip:
    """
    Hover (i) tooltip. Each call to SwingAnalysisCanvas.plot() creates a NEW
    PanelTooltip which disconnects the previous one so stale handlers don't accumulate.
    """
    RADIUS_PX=22
    def __init__(self,canvas):
        self.canvas=canvas; self._items=[]; self._cid=None
        self._cid=canvas.mpl_connect('motion_notify_event',self._on_hover)
    def disconnect(self):
        try:
            if self._cid is not None: self.canvas.mpl_disconnect(self._cid)
        except Exception: pass
    def register(self,ax,info_str):
        # Place (i) at bottom-right of axes so it doesn't overlap chart data
        sym=ax.text(0.995,0.02,'(i)',transform=ax.transAxes,ha='right',va='bottom',
                    color=INFO_C,fontsize=9,fontweight='bold',zorder=20,alpha=0.90,
                    bbox=dict(boxstyle='round,pad=0.2',fc='#1a2a4a',ec=INFO_C,alpha=0.7))
        annot=ax.annotate(info_str,xy=(0.995,0.02),xycoords='axes fraction',
                          xytext=(-8,18),textcoords='offset points',
                          ha='right',va='bottom',fontsize=8.5,color='white',
                          bbox=dict(boxstyle='round,pad=0.5',fc='#1a2a4a',ec=INFO_C,alpha=0.96),
                          zorder=30,visible=False)
        self._items.append((sym,annot))
    def _on_hover(self,event):
        if event.x is None or event.y is None: return
        redraw=False
        for sym,annot in self._items:
            ax=sym.axes
            want=False
            if ax is not None and event.inaxes==ax:
                try:
                    disp=ax.transAxes.transform((0.995,0.02))
                    want=np.hypot(event.x-disp[0],event.y-disp[1])<self.RADIUS_PX
                except Exception: pass
            if want!=annot.get_visible(): annot.set_visible(want); redraw=True
        if redraw:
            try: self.canvas.draw_idle()
            except: pass


class Skeleton3DCanvas(FigureCanvas):
    frame_changed=pyqtSignal(int); SCALE=0.38
    def __init__(self):
        self.fig=Figure(figsize=(5,4),facecolor=DARK); super().__init__(self.fig)
        self._azim=-70; self._side=False; self.user_seq=self.pro_seq=None
        self.fps=30; self.cur=0; self._pro_name='Pro'
        self._timer=QTimer(); self._timer.timeout.connect(self._advance)
    def _ax_s(self,ax,title=""):
        s=self.SCALE; ax.set_facecolor(DARK)
        for p in[ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]: p.fill=False; p.set_edgecolor(ACCENT)
        ax.set_xlim(-2.2*s,2.2*s); ax.set_ylim(-0.9*s,0.9*s); ax.set_zlim(-2.8*s,2.0*s)
        ax.set_xlabel("L/R",color=AXIS_C,fontsize=7); ax.set_ylabel("Depth",color=AXIS_C,fontsize=7)
        ax.set_zlabel("Height",color=AXIS_C,fontsize=7)
        for a in[ax.xaxis,ax.yaxis,ax.zaxis]: a.label.set_color(AXIS_C); a.set_tick_params(labelcolor=AXIS_C,labelsize=6)
        try: ax.set_box_aspect([4.4,1.8,4.8])
        except: pass
        ax.view_init(elev=18,azim=self._azim)
        if title: ax.set_title(title,color='white',fontsize=10,pad=3)
    def set_sequences(self,u,p,fps=30,user_phases=None,pro_phases=None,pro_name='Pro'):
        self.user_seq=u; self.pro_seq=p; self.fps=fps; self.cur=0
        self._pro_name=pro_name
        self._draw(0)
    def toggle_side(self): self._side=not self._side; self._draw(self.cur); return self._side
    def rotate(self,d): self._azim+=d; self._draw(self.cur)
    def play(self): self._timer.start(33)
    def pause(self): self._timer.stop()
    def set_frame(self,pct): self.cur=int(np.clip(pct,0,100)); self._draw(self.cur)
    def _advance(self):
        if self.user_seq is not None and len(self.user_seq):
            self.cur=(self.cur+1)%101; self._draw(self.cur); self.frame_changed.emit(self.cur)
    def _remap(self,p): s=self.SCALE; return float(p[0]*s),float(p[2]*s),float(-p[1]*s)
    def _draw_skel(self,ax,lm,color,label,alpha=1.0):
        first=True
        for si,ei in SKELETON_CONNECTIONS:
            if si>=len(lm) or ei>=len(lm): continue
            sx,sy,sz=self._remap(lm[si]); ex,ey,ez=self._remap(lm[ei])
            lc='#00ffdd' if(si in RIGHT_ARM or ei in RIGHT_ARM) else '#ff88cc' if(si in LEFT_ARM or ei in LEFT_ARM) else color
            ax.plot([sx,ex],[sy,ey],[sz,ez],color=lc,alpha=alpha,linewidth=1.8)
        for idx in KEY_LANDMARKS:
            if idx>=len(lm): continue
            px,py,pz=self._remap(lm[idx])
            ax.scatter(px,py,pz,color=color,s=14,alpha=alpha,depthshade=False,
                       label=label if first else ''); first=False
        if 14<len(lm) and 16<len(lm):
            ex,ey,ez=self._remap(lm[14]); wx,wy,wz=self._remap(lm[16])
            dx,dy,dz=wx-ex,wy-ey,wz-ez; d=np.sqrt(dx**2+dy**2+dz**2)+1e-8; sh=0.22*self.SCALE
            hx=wx+dx/d*sh; hy=wy+dy/d*sh; hz=wz+dz/d*sh
            ax.plot([wx,hx],[wy,hy],[wz,hz],color=YELLOW,lw=3,alpha=alpha,zorder=10)
            ax.scatter(hx,hy,hz,color=YELLOW,s=40,alpha=alpha,depthshade=False,zorder=10)
    def _draw(self,pct):
        f=int(np.clip(pct,0,100)); pn=self._pro_name
        self.fig.clear()
        if self._side:
            ax1=self.fig.add_subplot(121,projection='3d'); ax2=self.fig.add_subplot(122,projection='3d')
            self._ax_s(ax1); self._ax_s(ax2)
            ax1.set_title(f"You ({pct}%)",color=BLUE,fontsize=10,pad=3)
            ax2.set_title(f"{pn} ({pct}%)",color=PRO_COLOR,fontsize=10,pad=3)
            if self.user_seq is not None and len(self.user_seq):
                self._draw_skel(ax1,self.user_seq[f],BLUE,'You')
            if self.pro_seq is not None and len(self.pro_seq):
                self._draw_skel(ax2,self.pro_seq[f],PRO_COLOR,pn)
        else:
            ax=self.fig.add_subplot(111,projection='3d'); self._ax_s(ax)
            if self.user_seq is not None and len(self.user_seq):
                self._draw_skel(ax,self.user_seq[f],BLUE,'You')
            if self.pro_seq is not None and len(self.pro_seq):
                self._draw_skel(ax,self.pro_seq[f],PRO_COLOR,pn,0.50)
            ax.legend(facecolor=DARK,labelcolor='white',loc='upper right',fontsize=8)
        self.fig.canvas.draw_idle()


class OverlayCanvas(FigureCanvas):
    def __init__(self):
        self.fig=Figure(figsize=(7,5),facecolor=DARK); super().__init__(self.fig)
        self._side=False; self._pro_name='Pro'
    def toggle_side(self): self._side=not self._side; return self._side
    def set_pro_name(self,name): self._pro_name=name
    def draw_overlay(self,u,p):
        pn=self._pro_name
        self.fig.clear()
        if self._side:
            for i,(lm,clr,lbl) in enumerate([(u,BLUE,'You'),(p,PRO_COLOR,pn)]):
                ax=self.fig.add_subplot(1,2,i+1,facecolor='#0d0d1a'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
                ax.set_title(lbl,color=clr,fontsize=11,fontweight='bold')
                if lm is not None: self._sk(ax,lm,clr); self._rack(ax,lm)
        else:
            ax=self.fig.add_subplot(111,facecolor='#0d0d1a'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
            for lm,clr,lbl in[(u,BLUE,'You'),(p,PRO_COLOR,pn)]:
                if lm is not None: self._sk(ax,lm,clr,lbl); self._rack(ax,lm)
            ax.legend(handles=[mpatches.Patch(color=BLUE,label='You'),
                                mpatches.Patch(color=PRO_COLOR,label=pn),
                                mpatches.Patch(color=YELLOW,label='Racket')],
                      facecolor=DARK,labelcolor='white',loc='upper right',fontsize=8)
        self.fig.canvas.draw_idle()
    def _sk(self,ax,lm,clr,lbl='',show_lbl=True):
        for s,e in SKELETON_CONNECTIONS:
            if s<len(lm) and e<len(lm): ax.plot([lm[s,0],lm[e,0]],[lm[s,1],lm[e,1]],color=clr,lw=2,alpha=0.9)
        for idx in KEY_LANDMARKS:
            if idx<len(lm): ax.scatter(lm[idx,0],lm[idx,1],color=clr,s=18,zorder=5)
        if show_lbl and len(lm)>0: ax.text(lm[0,0],lm[0,1]+0.05,lbl,color=clr,fontsize=9,ha='center',fontweight='bold')
    def _rack(self,ax,lm):
        if lm is None or 14>=len(lm) or 16>=len(lm): return
        ex,ey=lm[14,0],lm[14,1]; wx,wy=lm[16,0],lm[16,1]
        dx,dy=wx-ex,wy-ey; dist=np.hypot(dx,dy)+1e-8; dx/=dist; dy/=dist
        shaft=0.065; ha=0.028; hb=0.020; tx=wx+dx*shaft; ty=wy+dy*shaft
        ax.plot([wx,tx],[wy,ty],color=YELLOW,lw=2.5,zorder=9)
        px,py=-dy,dx; t_=np.linspace(0,2*np.pi,28)
        ax.fill(tx+hb*np.cos(t_)*px+ha*np.sin(t_)*dx,ty+hb*np.cos(t_)*py+ha*np.sin(t_)*dy,color=YELLOW,alpha=0.20,zorder=8)
        ax.plot(tx+hb*np.cos(t_)*px+ha*np.sin(t_)*dx,ty+hb*np.cos(t_)*py+ha*np.sin(t_)*dy,color=YELLOW,lw=1.8,zorder=9)


class JointAnalysisCanvas(FigureCanvas):
    def __init__(self):
        self.fig=Figure(figsize=(12,16),facecolor=DARK); super().__init__(self.fig)
        self.setMinimumHeight(900)
    def plot(self,errors,pro_name='Pro'):
        self.fig.clear(); joints=list(errors.keys()); n=len(joints)
        if not n: return
        rows=(n+1)//2; cols=2
        for i,joint in enumerate(joints):
            ax=self.fig.add_subplot(rows,cols,i+1,facecolor='#0d0d1a')
            u=errors[joint]['user_arr']; p=errors[joint]['pro_arr']; x_=np.linspace(0,100,len(u))
            ax.plot(x_,u,color=BLUE,lw=2,label='You'); ax.plot(x_,p,color=PRO_COLOR,lw=2,label=pro_name,alpha=0.9)
            jdesc=JOINT_DESCRIPTIONS.get(joint,(joint.replace('_',' ').title(),''))
            # Title with hint on second line
            hint_short=jdesc[1].split('\n')[0] if jdesc[1] else ''
            ax.set_title(f"{jdesc[0]}\n{hint_short}",color='white',fontsize=8.5,fontweight='bold',pad=6,
                         multialignment='center',linespacing=1.6)
            ax.set_xlabel('Stroke Progress (%)',color=AXIS_C,fontsize=7)
            ax.set_ylabel('Angle (deg)',color=AXIS_C,fontsize=7)
            ax.tick_params(colors=AXIS_C,labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor('#333')
            ax.legend(facecolor=DARK,labelcolor='white',fontsize=7,loc='upper right')
            err=errors[joint]['mean_error']; clr='#ff6666' if err>15 else '#ffaa00' if err>8 else GREEN
            ax.text(0.02,0.03,f'Avg diff: {err:.1f} deg',transform=ax.transAxes,
                    ha='left',va='bottom',color=clr,fontsize=8)
        self.fig.subplots_adjust(left=0.08,right=0.97,top=0.97,bottom=0.04,hspace=0.85,wspace=0.38)
        self.fig.patch.set_facecolor(DARK); self.fig.canvas.draw_idle()


class SwingAnalysisCanvas(FigureCanvas):
    def __init__(self):
        self.fig=Figure(figsize=(11,14),facecolor=DARK,tight_layout=False)
        super().__init__(self.fig); self.setMinimumHeight(1100)
        self._show_pro=True; self._tip=None
    def set_show_pro(self,v): self._show_pro=v
    def plot(self,user_lm3d,phases_u,phases_p,phases_u_s,phases_p_s,
             rs_real,pro_rs_real,peak_rs,rs_cont,contact_label,contact_diff,
             rack_ang,stroke_type,sf_u,sf_p,fps,errors=None,pro_name='Pro'):
        # Disconnect old tooltip before clearing figure (stale axes refs)
        if self._tip is not None: self._tip.disconnect()
        self.fig.clear()
        T=len(user_lm3d)
        pdefs=STROKE_PHASES.get(stroke_type,STROKE_PHASES['forehand'])
        pkeys=[p[0] for p in pdefs]; pcolrs=[p[2] for p in pdefs]
        gs=gridspec.GridSpec(4,2,figure=self.fig,height_ratios=[3.2,1.6,1.6,3.0],hspace=0.68,wspace=0.38)
        gs.update(left=0.08,right=0.97,top=0.97,bottom=0.02)
        self._tip=PanelTooltip(self)

        # ── Speed graph: x-axis = stroke progress 0-100% ─────────────
        ax1=self.fig.add_subplot(gs[0,:],facecolor='#0d0d1a')
        # Normalise user to 0-100% of their frame count
        t_r=np.linspace(0,100,T)
        ax1.plot(t_r,rs_real,color='#ff9900',lw=2.5,label='You',zorder=3)
        if self._show_pro and pro_rs_real is not None:
            T_p=len(pro_rs_real)
            t_p=np.linspace(0,100,T_p)
            # Scale pro curve to user's peak so both curves are visually comparable.
            # Raw sw/s values are fps-dependent and can't be directly compared.
            u_peak=float(np.max(rs_real)) if np.max(rs_real)>0 else 1.0
            p_peak=float(np.max(pro_rs_real)) if np.max(pro_rs_real)>0 else 1.0
            pro_scaled=pro_rs_real*(u_peak/p_peak)
            ax1.plot(t_p,pro_scaled,color=BLUE,lw=1.8,alpha=0.7,linestyle='--',label=f'{pro_name} (scaled)',zorder=2)
            # Pro phase markers (lighter, dotted) from frame-based phases_p
            for k,col in zip(pkeys,pcolrs):
                if k in phases_p and T_p>1:
                    pct_p=phases_p[k]/max(T_p-1,1)*100
                    ax1.axvline(pct_p,color=col,lw=0.8,alpha=0.40,linestyle=':')
        # User phase markers
        for k,col in zip(pkeys,pcolrs):
            if k in phases_u and T>1:
                pct_u=phases_u[k]/max(T-1,1)*100
                ax1.axvline(pct_u,color=col,lw=1.6,alpha=0.85,linestyle='--')
                ax1.text(pct_u+0.5,ax1.get_ylim()[0] if ax1.get_ylim()[0]>0 else 0,
                         k.replace('_',' ').split(' ')[-1][:4],
                         color=col,fontsize=6,va='bottom',rotation=90,alpha=0.7)
        cf_pct=phases_u.get('contact',0)/max(T-1,1)*100
        rs_at=rs_cont
        ax1.axvline(cf_pct,color=YELLOW,lw=2.5)
        ax1.annotate(f"Contact\n{rs_cont:.1f} sw/s\n({rs_cont*SW_METERS*3.6:.0f} km/h)",
                     xy=(cf_pct,rs_at),xytext=(min(cf_pct+3,85),rs_at*0.70),
                     color=YELLOW,fontsize=8.5,ha='left',
                     arrowprops=dict(arrowstyle='->',color=YELLOW,lw=1.2))
        ax1.set_title('Racket Head Speed  (stroke progress %)',color='white',fontsize=12,pad=5)
        ax1.set_xlabel('Stroke Progress (%)',color=AXIS_C,fontsize=9)
        ax1.set_ylabel('Speed (sw/s)',color=AXIS_C,fontsize=9)
        ax1.set_xlim(0,100)
        ax1.tick_params(colors=AXIS_C,labelsize=8)
        for sp in ax1.spines.values(): sp.set_edgecolor('#333')
        ax1.text(0.99,0.97,f"Peak: {peak_rs:.1f} sw/s  ({peak_rs*SW_METERS*3.6:.0f} km/h)",
                 transform=ax1.transAxes,ha='right',va='top',color=GREEN,fontsize=11,fontweight='bold')
        if sf_u<0.75:
            ax1.text(0.01,0.97,f"Slow-mo ({sf_u:.3f}x) -- corrected",
                     transform=ax1.transAxes,ha='left',va='top',color='#ffaa44',fontsize=8.5)
        ax1.legend(facecolor=DARK,labelcolor='white',fontsize=9,loc='upper left',ncol=2,
                   bbox_to_anchor=(0.01,0.92),framealpha=0.7)
        self._tip.register(ax1,PANEL_INFO['speed'])

        # Phase bars
        ax2=self.fig.add_subplot(gs[1,:],facecolor='#0d0d1a')
        self._phase_row(ax2,pdefs,phases_u_s,'YOU',BLUE,title=f'Phase Durations -- You vs {pro_name}  [real seconds]')
        self._tip.register(ax2,PANEL_INFO['phases'])
        ax3=self.fig.add_subplot(gs[2,:],facecolor='#0d0d1a')
        self._phase_row(ax3,pdefs,phases_p_s,pro_name.upper(),PRO_COLOR,title='')
        self._tip.register(ax3,PANEL_INFO['phases'])
        ax4=self.fig.add_subplot(gs[3,0],facecolor='#0d0d1a')
        self._contact_diag(ax4,contact_diff,contact_label)
        self._tip.register(ax4,PANEL_INFO['contact'])
        ax5=self.fig.add_subplot(gs[3,1],facecolor='#0d0d1a')
        _,out_k=estimate_ball_speeds(rs_cont,stroke_type)
        self._metrics(ax5,phases_u_s,phases_p_s,pdefs,peak_rs,rs_cont,out_k,contact_label,rack_ang,sf_u)
        self._tip.register(ax5,PANEL_INFO['metrics'])
        self.fig.patch.set_facecolor(DARK); self.fig.canvas.draw_idle()
    def _phase_row(self,ax,pdefs,phases_s,row_lbl,row_clr,title=''):
        ax.set_facecolor('#0d0d1a'); ax.axis('off')
        if title: ax.set_title(title,color='white',fontsize=11,pad=4)
        keys=[p[0] for p in pdefs]; names=[p[1] for p in pdefs]; cols=[p[2] for p in pdefs]
        times=[phases_s.get(k,np.nan) for k in keys]; finite=[t for t in times if np.isfinite(t)]
        if not finite: return
        t0_all=min(finite); span=max(max(finite)-t0_all,0.01)
        norm=lambda t:(t-t0_all)/span if np.isfinite(t) else 0.0
        ax.text(-0.006,0.45,row_lbl,ha='right',va='center',color=row_clr,fontsize=10,fontweight='bold',transform=ax.transAxes)
        for i in range(len(keys)-1):
            t0=times[i]; t1=times[i+1]
            if not(np.isfinite(t0) and np.isfinite(t1)): continue
            dt=t1-t0; x0=norm(t0); x1=norm(t1); w=x1-x0; mx=(x0+x1)/2
            rect=mpatches.FancyBboxPatch((x0,0.12),w,0.76,boxstyle="round,pad=0.004",
                facecolor=cols[i+1],alpha=0.83,transform=ax.transAxes,clip_on=True)
            ax.add_patch(rect)
            if w>=0.07:
                ax.text(mx,0.50,f"{names[i+1]}\n{dt:.2f}s",ha='center',va='center',
                        color='white',fontsize=8.5,fontweight='bold',transform=ax.transAxes)
            else:
                ax.annotate(f"{names[i+1]}\n{dt:.2f}s",xy=(mx,0.88),xycoords='axes fraction',
                    xytext=(mx,1.12),textcoords='axes fraction',ha='center',va='bottom',
                    color=cols[i+1],fontsize=7.5,fontweight='bold',
                    arrowprops=dict(arrowstyle='-',color=cols[i+1],lw=1.2,shrinkA=0,shrinkB=0))
    def _contact_diag(self,ax,cd,cl):
        ax.set_xlim(-2,2); ax.set_ylim(-1.5,1.5); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title('Contact Point -- Top View\n(Where ball hit relative to body)',color='white',fontsize=10)
        ax.add_patch(mpatches.Circle((0,0),0.28,color='#445577',zorder=2))
        ax.text(0,0,'You',ha='center',va='center',color='white',fontsize=8,fontweight='bold',zorder=3)
        ax.annotate('',xy=(1.8,0),xytext=(0.35,0),arrowprops=dict(arrowstyle='->',color='#888888',lw=1.5))
        ax.text(1.87,0,'Fwd',color='#888888',fontsize=7,va='center')
        for x0,x1,lbl,clr,alp in[(-2.0,-0.30,'Behind\n(Late)','#ff4422',0.12),(-0.30,0.10,'Side','#ffaa22',0.12),(0.10,1.20,'Good Zone','#44aa44',0.12),(1.20,2.0,'Far Front','#2266aa',0.12)]:
            ax.add_patch(mpatches.Rectangle((x0,-1.3),x1-x0,2.6,facecolor=clr,alpha=alp,zorder=0))
            ax.text((x0+x1)/2,-1.25,lbl,ha='center',color=clr,fontsize=8)
        rx=float(np.clip(cd,-1.85,1.85))
        ax.scatter(rx,0.1,s=220,color=YELLOW,zorder=6,marker='D',edgecolors='white',linewidths=1.2)
        ax.text(rx,0.42,f'Racket\n{cl}',ha='center',color=YELLOW,fontsize=7.5,fontweight='bold',zorder=7)
    def _metrics(self,ax,pu,pp,pdefs,peak_rs,rs_cont,out_k,cl,ra,sf_u):
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
        ax.set_title('Key Metrics',color=BLUE,fontsize=13,fontweight='bold',pad=10)
        rows=[('Racket Head Speed',None,'H'),
              (f"  At contact: {rs_cont*SW_METERS:.1f} m/s  ({rs_cont*SW_METERS*3.6:.0f} km/h)",GREEN,'V'),
              (f"  Peak:        {peak_rs*SW_METERS:.1f} m/s  ({peak_rs*SW_METERS*3.6:.0f} km/h)",GREEN,'V'),
              ('','None','G'),('Ball Exit Speed  (physics proxy)',None,'H')]
        if out_k: rows+=[(f"  Outgoing: ~{out_k:.0f} km/h",YELLOW,'V')]
        rows+=[(f"  Incoming: requires ball tracker",AXIS_C,'S'),('','None','G'),
               (f"  Slow-motion factor: {sf_u:.3f}x",'#ffaa44' if sf_u<0.9 else GREEN,'S'),
               ('','None','G'),('Phase Durations  You vs Pro  (s)',None,'H')]
        for i in range(len(pdefs)-1):
            k0,k1=pdefs[i][0],pdefs[i+1][0]
            udt=pu.get(k1,np.nan)-pu.get(k0,np.nan); pdt=pp.get(k1,np.nan)-pp.get(k0,np.nan)
            if not(np.isfinite(udt) and np.isfinite(pdt)): continue
            rel=(udt-pdt)/max(pdt,0.01)
            fl='^' if rel>0.20 else('v' if rel<-0.20 else 'ok')
            rows+=[(f"  {pdefs[i+1][1]:12s}  {udt:.2f}s  (pro {pdt:.2f}s)  {fl}",pdefs[i+1][2],'V')]
        rows+=[('','None','G'),('Contact Analysis',None,'H'),
               (f"  Position: {cl}",YELLOW,'V'),(f"  Racket angle vs vertical: {ra:.1f} deg",YELLOW,'V')]
        y_=0.93; dH=0.060; dV=0.052; dS=0.042; dG=0.025
        for text,clr,kind in rows:
            if kind=='G': y_-=dG; continue
            if kind=='H': ax.text(0.01,y_,text,color=BLUE,fontsize=9.5,fontweight='bold',va='top'); y_-=dH
            elif kind=='S': ax.text(0.02,y_,text,color=clr or AXIS_C,fontsize=8.5,va='top'); y_-=dS
            else: ax.text(0.02,y_,text,color=clr or 'white',fontsize=9,va='top'); y_-=dV
            if y_<0.01: break


class BallTrajectoryCanvas(FigureCanvas):
    ELEV=18
    def __init__(self):
        self.fig=Figure(figsize=(40,12),facecolor=DARK)
        super().__init__(self.fig); self.setMinimumHeight(950)
        self._traj=None; self._stroke='forehand'; self._azim=-55
        self._wind=0.0; self._v0_rms=0.0; self._rack_ang=0.0; self._contact_h=0.90
        self._override_spin=None; self._override_launch=None; self._override_exit=None
        # Ball animation
        self._ball_t    = 0.0      # current physics time (seconds)
        self._ball_t_max= 0.0      # time cap: net-hit stops ball at net
        self._ball_times= None     # time→position lookup array (length of traj)
        self._ball_scat = None     # current scatter artist (removed/re-added each tick)
        self._ax        = None     # stored axes — reused across ticks (no full redraw)
        self._ball_timer= QTimer(); self._ball_timer.timeout.connect(self._ball_advance)

    def _build_time_array(self, traj):
        """Cumulative time for each traj point: physics step was dt=0.002 s."""
        return np.arange(len(traj['x'])) * 0.002

    def _ball_advance(self):
        """Timer tick: advance time, move ball sphere only (no full court redraw)."""
        if self._traj is None or self._ball_times is None or self._ax is None:
            self._ball_timer.stop(); return
        # Speed multiplier: real time passes at 1× (ball travels at true physics speed)
        self._ball_t += 0.016      # 16 ms per tick — keep in sync with 60fps timer
        if self._ball_t > self._ball_t_max:
            # Stop at net/end, pause 1 s, then replay
            self._ball_t = -1.0    # negative = pause phase
        if self._ball_t < 0.0:
            self._ball_t -= 0.016
            if self._ball_t < -0.8:   # pause for ~0.8 s then restart
                self._ball_t = 0.0
            return   # don't draw during pause
        self._move_ball()

    def _move_ball(self):
        """Remove old ball scatter, add new one at current time, partial redraw."""
        if self._ax is None or self._ball_times is None: return
        # Remove previous ball artist
        if self._ball_scat is not None:
            try: self._ball_scat.remove()
            except Exception: pass
            self._ball_scat = None
        t_cur = float(np.clip(self._ball_t, 0.0, self._ball_times[-1]))
        xi = float(np.interp(t_cur, self._ball_times, self._traj['x']))
        yi = float(np.interp(t_cur, self._ball_times, self._traj['y']))
        zi = float(np.interp(t_cur, self._ball_times, self._traj['z']))
        if -2.0 <= xi <= COURT_LEN+6 and yi >= -0.05:
            self._ball_scat = self._ax.scatter(
                [xi],[zi],[yi],
                color='#ccee22', s=110, zorder=15,
                marker='o', edgecolors='#888800', linewidths=1.2,
                depthshade=False)
        self.fig.canvas.draw_idle()

    def set_trajectory(self,traj,stroke_type,v0_rms=0.0,rack_ang=0.0,contact_h=0.90,
                       override_spin=None,override_launch=None,override_exit=None):
        self._ball_timer.stop()
        self._traj=traj; self._stroke=stroke_type
        self._v0_rms=v0_rms; self._rack_ang=rack_ang; self._contact_h=contact_h
        self._override_spin=override_spin; self._override_launch=override_launch
        self._override_exit=override_exit
        self._ball_times = self._build_time_array(traj)
        self._ball_t = 0.0
        self._ball_scat = None; self._ax = None
        # Cap time at net if ball doesn't clear it
        if not traj.get('over_net', True):
            net_i = int(np.searchsorted(traj['x'], NET_X))
            net_i = min(net_i, len(self._ball_times)-1)
            self._ball_t_max = float(self._ball_times[net_i])
        else:
            self._ball_t_max = float(self._ball_times[-1])
        try: self._draw()
        except Exception: import traceback; traceback.print_exc()
        self._ball_timer.start(16)   # ~60fps timer

    def set_azim(self,azim):
        self._azim=azim; self._ax=None; self._ball_scat=None
        try: self._draw()
        except Exception: pass

    def set_wind(self,wind_ms):
        self._wind=float(wind_ms)
        if self._v0_rms>0:
            try:
                self._traj=compute_ball_trajectory_physics(
                    self._v0_rms,self._rack_ang,self._stroke,self._contact_h,self._wind,
                    override_spin_rpm=self._override_spin,
                    override_launch_deg=self._override_launch,
                    override_exit_kmh=self._override_exit)
                self._ball_times=self._build_time_array(self._traj)
                self._ball_t=0.0; self._ball_scat=None; self._ax=None
                if not self._traj.get('over_net', True):
                    net_i = int(np.searchsorted(self._traj['x'], NET_X))
                    self._ball_t_max = float(self._ball_times[min(net_i,len(self._ball_times)-1)])
                else:
                    self._ball_t_max = float(self._ball_times[-1])
            except Exception: pass
        try: self._draw()
        except Exception: pass
    def _draw(self):
        self._ball_timer.stop()
        self._ball_scat = None   # old ax is gone after fig.clear()
        self.fig.clear()
        ax=self.fig.add_subplot(111,projection='3d')
        self._ax = ax              # store for ball-only updates
        ax.set_facecolor(DARK)
        for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
            pane.fill=False; pane.set_edgecolor('none')
        try:
            for info in [ax.xaxis._axinfo,ax.yaxis._axinfo,ax.zaxis._axinfo]:
                info['grid']['linewidth']=0; info['grid']['color']=(0,0,0,0)
        except Exception: pass
        ax.view_init(elev=self.ELEV,azim=self._azim)
        # set_box_aspect with small safe integers makes the 3D box elongated
        # (court x >> court y >> height) — keeps court proportions visually correct
        # without crashing matplotlib on Windows (large numbers crash; [2,1,1] is safe)
        try: ax.set_box_aspect([2, 1, 1])
        except Exception: pass
        cw2=DBL_W/2; sw2=SGL_W/2
        verts=[[(0,-cw2,0),(COURT_LEN,-cw2,0),(COURT_LEN,cw2,0),(0,cw2,0)]]
        ax.add_collection3d(Poly3DCollection(verts,alpha=0.15,facecolor='#1a4a1a',edgecolor='none'))
        lm='#33aa33'; lt='#226622'
        for bx in [0,COURT_LEN]: ax.plot([bx,bx],[-cw2,cw2],[0,0],color=lm,lw=2.8)
        for sz in [-cw2,cw2]: ax.plot([0,COURT_LEN],[sz,sz],[0,0],color=lm,lw=1.8)
        for sz in [-sw2,sw2]: ax.plot([0,COURT_LEN],[sz,sz],[0,0],color=lt,lw=1.2)
        for sx in [SVC_LINE_P,SVC_LINE_O]: ax.plot([sx,sx],[-sw2,sw2],[0,0],color=lt,lw=1.0)
        ax.plot([SVC_LINE_P,SVC_LINE_O],[0,0],[0,0],color=lt,lw=1.0)
        for bx in [0,COURT_LEN]: ax.plot([bx,bx],[-0.20,0.20],[0,0],color=lt,lw=1.0)
        z_net=np.linspace(-cw2,cw2,60)
        h_net=NET_H_CTR+(NET_H_POST-NET_H_CTR)*(2*np.abs(z_net)/DBL_W)**2
        ax.plot([NET_X]*60,z_net,h_net,color='white',lw=2.5,alpha=0.90,zorder=6)
        for z_s in np.linspace(-cw2,cw2,15):
            h_s=NET_H_CTR+(NET_H_POST-NET_H_CTR)*(2*abs(z_s)/DBL_W)**2
            ax.plot([NET_X,NET_X],[z_s,z_s],[0,h_s],color='white',lw=0.5,alpha=0.30)
        for z_p in [-cw2,cw2]: ax.plot([NET_X,NET_X],[z_p,z_p],[0,NET_H_POST],color='white',lw=2.0)
        ax.set_xlim(-2.0,COURT_LEN+2.0); ax.set_ylim(-cw2-0.5,cw2+0.5); ax.set_zlim(0,10)
        ax.set_xlabel('Court (m)',color=AXIS_C,fontsize=10,labelpad=4)
        ax.set_ylabel('Width (m)',color=AXIS_C,fontsize=10,labelpad=4)
        ax.set_zlabel('Height (m)',color=AXIS_C,fontsize=10,labelpad=4)
        for a in [ax.xaxis,ax.yaxis,ax.zaxis]: a.label.set_color(AXIS_C); a.set_tick_params(labelcolor=AXIS_C,labelsize=8)
        # No set_box_aspect -- it forces a cube that kills the wide rectangle shape
        ax.scatter([-1.0],[0],[0],color=BLUE,s=130,zorder=8,marker='^',edgecolors='white',linewidths=1)
        ax.text(-1.0,0,0.4,'Player',color=BLUE,fontsize=10,ha='center')
        if self._traj is not None:
            try: self._draw_traj(ax,self._traj)
            except Exception: import traceback; traceback.print_exc()
        else:
            ax.text2D(0.5,0.5,'Analyse a stroke to see ball trajectory',
                      transform=ax.transAxes,ha='center',va='center',color=AXIS_C,fontsize=13)
        if abs(self._wind)>0.05:
            wlbl=f"Tailwind +{self._wind:.1f} m/s" if self._wind>0 else f"Headwind {self._wind:.1f} m/s"
            ax.text2D(0.50,0.97,wlbl,transform=ax.transAxes,ha='center',va='top',
                      color='#88ddff',fontsize=11,fontweight='bold')
        ax.set_title('Ball Trajectory -- 3D Court  (rotate & wind sliders below)',color='white',fontsize=11,pad=2)
        # Push axes to fill the full wide figure rectangle with minimal top gap
        self.fig.subplots_adjust(left=0.01,right=0.99,top=0.985,bottom=0.02)
        self.fig.patch.set_facecolor(DARK); self.fig.canvas.draw_idle()
        # Draw initial ball position, then restart animation timer
        self._move_ball()
        if self._ball_times is not None:
            self._ball_timer.start(16)

    def _find_bounces(self,xa,ya):
        """Find indices where ball y-crosses from airborne to ground."""
        bounces=[]
        for i in range(1,len(ya)):
            if xa[i]>0.5 and ya[i]<=0.06 and ya[i-1]>0.06:
                bounces.append(i)
                if len(bounces)==2: break
        return bounces

    def _draw_traj(self,ax,traj):
        x=traj['x']; y=traj['y']; z=traj['z']
        bounces=self._find_bounces(x,y)
        b1=bounces[0] if len(bounces)>=1 else None

        if not traj['over_net']:
            # Ball hit the net. Show the FULL trajectory from launch to net,
            # including any bounce on the server's side, then stop at the net.
            # Find net crossing index in the FULL array (not clipped at b1).
            net_i_full = int(np.searchsorted(x, NET_X))
            net_i_full = min(net_i_full, len(x)-1)
            xa_to_net = x[:net_i_full+1]; ya_to_net = y[:net_i_full+1]; za_to_net = z[:net_i_full+1]

            if b1 is not None and b1 < net_i_full:
                # Server-side bounce exists — show approach + bounce arc separately
                ax.plot(x[:b1+1], z[:b1+1], y[:b1+1],
                        color='#44aaff', lw=3.5, zorder=5, label='Approach')
                ax.plot(x[:b1+1], z[:b1+1], np.zeros(b1+1),
                        color='white', lw=0.8, alpha=0.20, linestyle=':')
                # Post-bounce arc to net in red
                ax.plot(x[b1:net_i_full+1], z[b1:net_i_full+1], y[b1:net_i_full+1],
                        color=RED, lw=3.0, zorder=5, linestyle='--', label='Into net')
            else:
                ax.plot(xa_to_net, za_to_net, ya_to_net,
                        color=RED, lw=3.5, zorder=5, label='Hit net')
                ax.plot(xa_to_net, za_to_net, np.zeros_like(xa_to_net),
                        color='white', lw=0.8, alpha=0.20, linestyle=':')
            # Red X marker at net
            ax.scatter([NET_X],[0],[traj['y_net']], color=RED, s=300, zorder=12,
                       marker='x', linewidths=3)
            ax.text(NET_X+0.3, 0, traj['y_net']+0.25,
                    'Hit Net', color=RED, fontsize=9, fontweight='bold')
        else:
            # Ball over net
            end1=b1 if b1 is not None else len(x)
            xb=x[:end1+1]; yb=y[:end1+1]; zb=z[:end1+1]
            net_i=int(np.searchsorted(xb,NET_X))
            if 0<net_i<len(xb):
                ax.plot(xb[:net_i+1],zb[:net_i+1],yb[:net_i+1],
                        color='#44aaff',lw=3.5,zorder=5,label='Approach')
                ax.plot(xb[net_i:],zb[net_i:],yb[net_i:],
                        color=GREEN,lw=3.5,zorder=5,label='After net')
            else:
                ax.plot(xb,zb,yb,color='#44aaff',lw=3.5,zorder=5,label='Arc')
            ax.plot(xb,zb,np.zeros_like(xb),color='white',lw=0.8,alpha=0.20,linestyle=':')
            # Apex marker
            if len(yb)>3:
                ai=int(np.argmax(yb))
                ax.scatter([xb[ai]],[zb[ai]],[yb[ai]],color='#ffaa00',s=80,zorder=8,
                           marker='^',edgecolors='white',linewidths=0.8)
                ax.text(xb[ai],zb[ai],yb[ai]+0.25,f"Peak\n{yb[ai]:.1f}m",
                        color='#ffaa00',fontsize=8,ha='center')
            # Post-bounce arc (first bounce only)
            if b1 is not None:
                xc=x[b1:]; yc=y[b1:]; zc=z[b1:]
                xc=xc[:min(len(xc),int(len(xb)*0.6)+1)]
                yc=yc[:len(xc)]; zc=zc[:len(xc)]
                if len(xc)>2:
                    ax.plot(xc,zc,yc,color='#ffcc44',lw=2.5,linestyle='--',
                            zorder=5,label='Post-bounce',alpha=0.92)

        # Landing star (only if over net)
        if traj['over_net']:
            lx=traj['x_land']; lz=traj['z_land']; in_c=traj['in_court']
            lclr=GREEN if in_c else RED
            ax.scatter([lx],[lz],[0],color=lclr,s=280,zorder=10,
                       marker='*',edgecolors='white',linewidths=1.2)
            st=traj.get('stroke_type','forehand')
            in_label='IN service box' if(in_c and st=='serve') else('IN' if in_c else 'OUT')
            ax.text(lx,lz,0.45,f"{in_label}\n{lx:.1f}m",
                    color=lclr,fontsize=9,fontweight='bold',ha='center')

        # Net clearance diamond
        y_net=traj['y_net']; net_clr=GREEN if traj['over_net'] else RED
        clearance=y_net-NET_H_CTR
        ax.scatter([NET_X],[0],[y_net],color=net_clr,s=80,zorder=9,
                   marker='D',edgecolors='white',linewidths=0.8)
        clr_txt=(f"+{clearance:.2f}m" if clearance>=0 else f"{clearance:.2f}m")
        ax.text(NET_X+0.4,0,y_net+0.28,f"Net cl: {clr_txt}",
                color=net_clr,fontsize=8,ha='left')

        spin_rpm=traj.get('spin_RPM',0); wind_ms=traj.get('wind_ms',0.0)
        st=traj.get('stroke_type','forehand')
        w_line=(f"Wind: {wind_ms:+.1f} m/s\n") if abs(wind_ms)>0.05 else ""
        if traj['over_net']:
            lx=traj['x_land']; in_c=traj['in_court']
            in_label='IN service box' if(in_c and st=='serve') else('IN' if in_c else 'OUT')
            land_line=f"Lands:   {lx:.1f} m  ({in_label})\n"
        else:
            land_line="Lands:   Hit net\n"
        stats=(f"Stroke: {st.capitalize()}\nLaunch:  {traj['launch_deg']:.1f} deg\n"
               f"Exit:      {traj['v0_ms']*3.6:.0f} km/h\nTopspin: {spin_rpm:.0f} RPM\n"
               f"{w_line}{land_line}Net cl:  {clearance:+.2f} m above net")
        ax.text2D(0.01,0.96,stats,transform=ax.transAxes,color='white',fontsize=9,va='top',
                  bbox=dict(boxstyle='round,pad=0.5',facecolor='#1a1a3a',alpha=0.88,edgecolor=ACCENT))
        ax.legend(facecolor=DARK,labelcolor='white',fontsize=8,loc='upper right',framealpha=0.75)


class VideoOverlayWidget(QWidget):
    """
    Shows user and pro video side-by-side or blended.

    Timing sync
    ───────────
    When analysis phases are provided, a piecewise-linear phase map is built
    (user_frame → pro_frame) so that both videos are at the same stroke phase
    at every point in time.  If phases are unavailable, falls back to a simple
    linear stretch.

    Overlay mode
    ────────────
    The pro frame is:
    1. Aspect-ratio-preserved (no squashing)
    2. Scaled so that the pro's shoulder width ≈ the user's shoulder width
    3. Translated so the pro's shoulder midpoint aligns with the user's
    """
    def __init__(self):
        super().__init__(); layout=QVBoxLayout(self)
        layout.setContentsMargins(4,4,4,4); layout.setSpacing(6)
        ctl=QHBoxLayout()
        self.btn_play=QPushButton("Play"); self.btn_play.setFixedWidth(100)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_mode=QPushButton("Overlay Mode"); self.btn_mode.setFixedWidth(160)
        self.btn_mode.clicked.connect(self._toggle_mode)
        self.lbl_info=QLabel("No video loaded")
        self.lbl_info.setStyleSheet(f"color:{AXIS_C};font-size:11px;")
        ctl.addWidget(self.btn_play); ctl.addWidget(self.btn_mode)
        ctl.addWidget(self.lbl_info); ctl.addStretch(); layout.addLayout(ctl)

        self.side_widget=QWidget()
        sl=QHBoxLayout(self.side_widget); sl.setContentsMargins(0,0,0,0); sl.setSpacing(4)
        self.lbl_user=QLabel("Your Video"); self.lbl_user.setAlignment(Qt.AlignCenter)
        self.lbl_user.setStyleSheet(
            f"background:#000;border:2px solid {BLUE};border-radius:4px;color:{AXIS_C};")
        self.lbl_user.setMinimumSize(300,220)
        self.lbl_pro=QLabel("Pro Video"); self.lbl_pro.setAlignment(Qt.AlignCenter)
        self.lbl_pro.setStyleSheet(
            f"background:#000;border:2px solid {PRO_COLOR};border-radius:4px;color:{AXIS_C};")
        self.lbl_pro.setMinimumSize(300,220)
        sl.addWidget(self.lbl_user); sl.addWidget(self.lbl_pro)
        layout.addWidget(self.side_widget)

        self.blend_widget=QWidget()
        bl=QVBoxLayout(self.blend_widget); bl.setContentsMargins(0,0,0,0)
        self.lbl_blend=QLabel("Overlay View"); self.lbl_blend.setAlignment(Qt.AlignCenter)
        self.lbl_blend.setStyleSheet(
            f"background:#000;border:1px solid {ACCENT};border-radius:4px;color:{AXIS_C};")
        self.lbl_blend.setMinimumSize(640,460)
        bl.addWidget(self.lbl_blend); self.blend_widget.hide()
        layout.addWidget(self.blend_widget)

        sr=QHBoxLayout(); sr.addWidget(QLabel("Frame:"))
        self.slider=QSlider(Qt.Horizontal); self.slider.setRange(0,100); sr.addWidget(self.slider)
        self.lbl_fn=QLabel("0"); self.lbl_fn.setFixedWidth(60)
        self.lbl_fn.setStyleSheet(f"color:{AXIS_C};"); sr.addWidget(self.lbl_fn)
        layout.addLayout(sr)

        self.user_frames=self.pro_frames=None
        self.user_lm2d=self.pro_lm2d=None
        self._phase_map=None     # user_frame_idx -> pro_frame_idx
        self._fixed_pro_scale=None   # computed once per set_videos call
        self._fixed_pro_size=None    # (new_pw, new_ph)
        self._ov=False; self._pl=False; self._fps=30.
        self._timer=QTimer(); self._timer.timeout.connect(self._advance)
        self.slider.valueChanged.connect(self._on_sl)

    # ── Load & setup ──────────────────────────────────────────────────
    def set_videos(self, up, pp, mapping_len=None, fps=30,
                   user_lm2d=None, pro_lm2d=None,
                   user_phases=None, pro_phases=None, stroke_type='forehand'):
        self._pl=False; self._timer.stop(); self.btn_play.setText("Play")
        self._fps=max(float(fps), 1.)
        self.user_frames=self._load(up)
        self.pro_frames =self._load(pp)
        self.user_lm2d=user_lm2d; self.pro_lm2d=pro_lm2d

        uf_len=len(self.user_frames) if self.user_frames else (mapping_len or 100)
        pf_len=len(self.pro_frames)  if self.pro_frames  else 1

        # Build independent 0-100 → frame-index maps for each video.
        # This decouples frame counts completely: both videos always play
        # from start to finish over the full 0-100 slider range.
        # Phase anchors ensure matching stroke moments are shown together.
        self._user_map = self._build_pct_map(user_phases, uf_len, stroke_type)  # pct→user frame
        self._pro_map  = self._build_pct_map(pro_phases,  pf_len, stroke_type)  # pct→pro frame

        # Slider is always 0-100 (stroke progress %)
        self.slider.setRange(0, 100); self.slider.setValue(0)
        u_n=uf_len; p_n=pf_len
        inf=f"User: {u_n} fr  |  Pro: {p_n} fr  [0-100% synced]"
        self.lbl_info.setText(inf)

        # Compute a FIXED pro scale using median shoulder widths across all frames.
        # We compute this once here so the pro stays the same size for every frame —
        # only its position changes to track the shoulder alignment.
        self._fixed_pro_scale = None
        self._fixed_pro_size  = None    # (new_pw, new_ph) for the pre-resized pro frame
        self._compute_fixed_scale()

        self._show(0)

    @staticmethod
    def _build_pct_map(phases, n_frames, stroke_type='forehand'):
        """Pure linear pct(0..100) → frame-index array. Always covers the full
        video from first to last loaded frame so both videos play completely."""
        if n_frames <= 0: return np.zeros(101, dtype=int)
        return np.clip(
            np.round(np.linspace(0, n_frames-1, 101)).astype(int),
            0, n_frames-1)

    def _compute_fixed_scale(self):
        """
        Compute a fixed pro scale by matching person height (head to feet).
        MediaPipe: landmark 0=nose, 27=left ankle, 28=right ankle.
        Scale = user_person_px_height / pro_person_px_height.
        Sampled across many frames, median taken for robustness.
        Same-video test: scale ≈ 1.0, overlay should near-perfectly align.
        """
        self._fixed_pro_scale = None
        self._fixed_pro_size  = None
        if not self.user_frames or not self.pro_frames: return
        uh, uw = self.user_frames[0].shape[:2]
        ph, pw = self.pro_frames[0].shape[:2]
        if uh<=0 or uw<=0 or ph<=0 or pw<=0: return

        def person_height_px(lm2d, frame_i, fw, fh):
            if lm2d is None or len(lm2d)==0: return None
            fi = int(np.clip(frame_i, 0, len(lm2d)-1))
            lm = lm2d[fi]
            if lm.shape[0] < 29: return None
            # Nose Y as head proxy, lower ankle Y as feet proxy
            nose_y = float(lm[0, 1]) * fh
            foot_y = max(float(lm[27,1])*fh, float(lm[28,1])*fh)
            h = foot_y - nose_y
            if h < fh*0.15 or h > fh*0.98: return None
            return h

        u_hs, p_hs = [], []
        for pct in range(5, 96, 5):
            uh_ = person_height_px(self.user_lm2d, self._uf(pct), uw, uh)
            ph_ = person_height_px(self.pro_lm2d,  self._pf(pct), pw, ph)
            if uh_ is not None: u_hs.append(uh_)
            if ph_ is not None: p_hs.append(ph_)

        if len(u_hs) >= 3 and len(p_hs) >= 3:
            scale = float(np.median(u_hs)) / float(np.median(p_hs))
        else:
            # Fallback: raw frame height ratio
            scale = uh / ph

        scale = float(np.clip(scale, 0.15, 3.5))
        new_pw = max(1, int(pw * scale))
        new_ph = max(1, int(ph * scale))
        self._fixed_pro_scale = scale
        self._fixed_pro_size  = (new_pw, new_ph)
        print(f"VideoOverlay: head-foot scale={scale:.3f} "
              f"({pw}x{ph}) → ({new_pw}x{new_ph})")

    def _load(self, path, mx=500):
        if not path: return None
        try:
            import pathlib; path=str(pathlib.Path(path).resolve())
            if not os.path.exists(path):
                print(f"VideoOverlay: not found: {path}"); return None
            cap=cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"VideoOverlay: cannot open: {path}"); return None
            fr=[]
            while len(fr)<mx:
                ret,f=cap.read()
                if not ret: break
                fr.append(f)
            cap.release()
            print(f"VideoOverlay: loaded {len(fr)} fr from {os.path.basename(path)}")
            return fr if fr else None
        except Exception as e:
            print(f"VideoOverlay _load: {e}"); return None

    # ── Frame index helpers ───────────────────────────────────────────
    def _uf(self, pct):
        """Get user frame index for stroke progress pct (0-100)."""
        if not self.user_frames: return 0
        pct=int(np.clip(pct, 0, 100))
        if hasattr(self,'_user_map') and self._user_map is not None:
            return int(self._user_map[pct])
        return min(int(pct*(len(self.user_frames)-1)//100), len(self.user_frames)-1)

    def _pf(self, pct):
        """Get pro frame index for stroke progress pct (0-100)."""
        if not self.pro_frames: return 0
        pct=int(np.clip(pct, 0, 100))
        if hasattr(self,'_pro_map') and self._pro_map is not None:
            return int(self._pro_map[pct])
        return min(int(pct*(len(self.pro_frames)-1)//100), len(self.pro_frames)-1)

    def _pi(self, i):
        """Legacy: map slider value (now pct) to pro frame."""
        return self._pf(i)

    # ── Playback ──────────────────────────────────────────────────────
    def _toggle_mode(self):
        if not self.user_frames:
            self.lbl_info.setText("Load a video and run analysis first."); return
        self._ov=not self._ov
        if self._ov:
            self.side_widget.hide(); self.blend_widget.show()
            self.btn_mode.setText("Side-by-Side")
        else:
            self.blend_widget.hide(); self.side_widget.show()
            self.btn_mode.setText("Overlay Mode")
        try: self._show(self.slider.value())
        except Exception: pass

    def _toggle_play(self):
        if self._pl:
            self._pl=False; self._timer.stop(); self.btn_play.setText("Play")
        else:
            if not self.user_frames: return
            self._pl=True
            # Always play at a fixed 30fps visual speed regardless of source fps.
            # Each tick advances 1 percentage point (101 ticks = full stroke).
            self._timer.start(33)   # 33ms ≈ 30fps
            self.btn_play.setText("Pause")

    def _advance(self):
        if not self.user_frames: return
        nv=(self.slider.value()+1) % 101   # 0..100
        self.slider.blockSignals(True); self.slider.setValue(nv)
        self.slider.blockSignals(False)
        self._show(nv); self.lbl_fn.setText(f"{nv}%")

    def _on_sl(self, v): self._show(v)

    def _show(self, pct):
        """Display frame at stroke progress pct (0-100)."""
        if not self.user_frames: return
        self.lbl_fn.setText(f"{pct}%")
        uf=self.user_frames[self._uf(pct)]
        pf=self.pro_frames[self._pf(pct)] if self.pro_frames else None
        if self._ov:
            try:
                fr=self._overlay(uf, pf, pct) if pf is not None else uf
                self._img(self.lbl_blend, fr if fr is not None else uf)
            except Exception as e:
                print(f"_show overlay error: {e}")
                self._img(self.lbl_blend, uf)
        else:
            self._img(self.lbl_user, uf)
            if pf is not None: self._img(self.lbl_pro, pf)
            else: self.lbl_pro.setText("No pro video available")

    # ── Overlay compositor ────────────────────────────────────────────
    def _overlay(self, uf, pf, pct):
        """
        Composite pro onto user canvas.
        - Scale is FIXED (computed once in _compute_fixed_scale)
        - Only translation changes per frame to track shoulder position
        """
        try:
            uh, uw = uf.shape[:2]
            ph, pw = pf.shape[:2]
            if uh<=0 or uw<=0 or ph<=0 or pw<=0: return uf

            # ── Get fixed pro size (computed once) ──────────────────
            if self._fixed_pro_size is None:
                # Fallback if scale was not computed yet
                new_pw = max(1, int(pw * 0.60))
                new_ph = max(1, int(ph * 0.60))
                scale  = 0.60
            else:
                new_pw, new_ph = self._fixed_pro_size
                scale = self._fixed_pro_scale

            # Resize pro to fixed size (always same dimensions every frame)
            pf_s = cv2.resize(pf, (new_pw, new_ph), interpolation=cv2.INTER_LINEAR)

            # ── Translation: align by hip midpoint ─────────────────
            # Hips (landmarks 23/24) are more stable than shoulders across
            # different stroke phases and camera angles.
            u_anchor = self._body_anchor(pct, 'user', uw, uh)
            p_anchor = self._body_anchor(pct, 'pro',  pw, ph)

            # Map pro anchor to scaled coordinates
            p_anchor_s = (p_anchor[0] * scale, p_anchor[1] * scale)

            tx = int(round(u_anchor[0] - p_anchor_s[0]))
            ty = int(round(u_anchor[1] - p_anchor_s[1]))

            # ── Paste onto canvas with 50% blend ────────────────────
            canvas = uf.copy().astype(np.float32)
            x0s = max(0, -tx);  y0s = max(0, -ty)
            x1s = min(new_pw, uw - tx); y1s = min(new_ph, uh - ty)
            x0d = max(0,  tx); y0d = max(0,  ty)
            x1d = x0d + (x1s - x0s); y1d = y0d + (y1s - y0s)

            if x1s > x0s and y1s > y0s and y1d <= uh and x1d <= uw:
                pro_patch = pf_s[y0s:y1s, x0s:x1s].astype(np.float32)
                canvas[y0d:y1d, x0d:x1d] = (
                    canvas[y0d:y1d, x0d:x1d] * 0.50 + pro_patch * 0.50)

            return canvas.clip(0, 255).astype(np.uint8)
        except Exception as e:
            print(f"_overlay: {e}"); return uf

    def _shoulder_pixels(self, pct, who, fw, fh):
        """
        Return (shoulder_mid_px, shoulder_width_px) at stroke progress pct.
        Falls back gracefully if landmarks are missing or zero.
        """
        lm2d=self.user_lm2d if who=='user' else self.pro_lm2d
        if lm2d is None or len(lm2d)==0:
            return (fw*0.5, fh*0.30), fw*0.20

        frame_i = self._uf(pct) if who=='user' else self._pf(pct)
        frame_i = int(np.clip(frame_i, 0, len(lm2d)-1))
        lm=lm2d[frame_i]

        # Try landmarks 11 (left shoulder) and 12 (right shoulder)
        if lm.shape[0] < 13:
            return (fw*0.5, fh*0.30), fw*0.20

        l_sh=lm[11,:2]; r_sh=lm[12,:2]
        # Check if landmarks are valid (not both zero, not NaN, within [0,1])
        valid = (not (np.all(l_sh==0) and np.all(r_sh==0)) and
                 np.all(np.isfinite(l_sh)) and np.all(np.isfinite(r_sh)) and
                 0<l_sh[0]<1 and 0<r_sh[0]<1)
        if not valid:
            return (fw*0.5, fh*0.30), fw*0.20

        mid_x=float((l_sh[0]+r_sh[0])/2)*fw
        mid_y=float((l_sh[1]+r_sh[1])/2)*fh
        sh_w=float(np.linalg.norm(l_sh-r_sh))*fw
        # Sanity check: shoulder width should be 5-50% of frame width
        if sh_w < fw*0.04 or sh_w > fw*0.60:
            return (fw*0.5, fh*0.30), fw*0.20
        return (mid_x, mid_y), sh_w

    # ── Image display ─────────────────────────────────────────────────
    def _body_anchor(self, pct, who, fw, fh):
        """
        Return (x_px, y_px) of the body anchor point used for translation.
        We use the hip midpoint (lm 23+24) which is stable across all stroke phases.
        Falls back to image centre if landmarks are invalid.
        """
        lm2d = self.user_lm2d if who=='user' else self.pro_lm2d
        if lm2d is None or len(lm2d)==0:
            return (fw*0.5, fh*0.55)
        frame_i = self._uf(pct) if who=='user' else self._pf(pct)
        frame_i = int(np.clip(frame_i, 0, len(lm2d)-1))
        lm = lm2d[frame_i]
        if lm.shape[0] < 25: return (fw*0.5, fh*0.55)
        lh = lm[23,:2]; rh = lm[24,:2]
        valid = (np.all(np.isfinite(lh)) and np.all(np.isfinite(rh)) and
                 0<lh[0]<1 and 0<rh[0]<1)
        if not valid: return (fw*0.5, fh*0.55)
        ax = float((lh[0]+rh[0])/2) * fw
        ay = float((lh[1]+rh[1])/2) * fh
        return (ax, ay)

    def _img(self, label, frame):
        try:
            if frame is None: return
            rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w,c=rgb.shape
            raw=bytes(rgb.tobytes())
            qi=QImage(raw, w, h, w*c, QImage.Format_RGB888)
            lw=label.width()  if label.width()>50  else 640
            lh=label.height() if label.height()>50 else 460
            pm=QPixmap.fromImage(qi).scaled(lw,lh,Qt.KeepAspectRatio,Qt.SmoothTransformation)
            label.setPixmap(pm)
        except Exception as e:
            print(f"_img: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Tennis Stroke Analyzer")
        self.setGeometry(80,60,1480,900); self.setStyleSheet(APP_STYLE)
        self.video_path=self.pro_vid_path=None; self.results=None; self.pro_cache={}
        self.current_pro_label=""; self._proc_thread=self._ai_thread=None
        self._vid_stopped_frame=0; self._vid_stopped_manually=False
        self.user_world=self.pro_world=None; self._overlay_playing=False
        self.user_world_r=self.pro_world_r=None   # trimmed+101-frame linear resampled
        self.pro_name='Pro'                        # actual pro name for labels
        self._overlay_timer=QTimer(); self._overlay_timer.timeout.connect(self._advance_overlay)
        self.base_prof_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data','professional'))
        self._build_ui(); self._load_pro_cache()

    def _build_ui(self):
        root=QWidget(); self.setCentralWidget(root); hbox=QHBoxLayout(root)
        left=QWidget(); left.setFixedWidth(310); lv=QVBoxLayout(left)
        t=QLabel("Tennis Stroke Analyzer"); t.setStyleSheet(f"font-size:16px;font-weight:bold;color:{RED};padding:6px;"); t.setAlignment(Qt.AlignCenter); lv.addWidget(t)
        g1=QGroupBox("1. Your Video"); v1=QVBoxLayout(g1)
        self.btn_upload=QPushButton("Select Video"); self.btn_upload.clicked.connect(self._pick_video)
        self.lbl_vid=QLabel("No video selected"); self.lbl_vid.setStyleSheet(f"color:{AXIS_C};font-size:11px;"); self.lbl_vid.setWordWrap(True)
        v1.addWidget(self.btn_upload); v1.addWidget(self.lbl_vid); lv.addWidget(g1)
        g2=QGroupBox("2. Stroke Type"); v2=QVBoxLayout(g2)
        self.combo_stroke=QComboBox(); self.combo_stroke.addItems(["Forehand","Backhand","Serve"])
        v2.addWidget(self.combo_stroke); lv.addWidget(g2)
        g3=QGroupBox("3. Pro Reference"); v3=QVBoxLayout(g3)
        self.btn_pro=QPushButton("Select Pro Video"); self.btn_pro.clicked.connect(self._pick_pro)
        self.lbl_pro=QLabel("Or use folder: data/professional/<stroke>/"); self.lbl_pro.setStyleSheet(f"color:{AXIS_C};font-size:11px;"); self.lbl_pro.setWordWrap(True)
        v3.addWidget(self.btn_pro); v3.addWidget(self.lbl_pro); lv.addWidget(g3)
        g4=QGroupBox("4. Video Speed"); v4=QVBoxLayout(g4)
        v4.addWidget(QLabel("Your video:"))
        self.combo_user_speed=QComboBox()
        for lbl,_ in SPEED_OPTIONS: self.combo_user_speed.addItem(lbl)
        v4.addWidget(self.combo_user_speed); v4.addWidget(QLabel("Pro video:"))
        self.combo_pro_speed=QComboBox()
        for lbl,_ in SPEED_OPTIONS: self.combo_pro_speed.addItem(lbl)
        v4.addWidget(self.combo_pro_speed)
        v4.addWidget(QLabel("Tip: set slow-mo factor if auto is wrong.")); lv.addWidget(g4)
        self.btn_analyze=QPushButton("Analyze Stroke")
        self.btn_analyze.setStyleSheet(f"background:{RED};font-size:14px;font-weight:bold;padding:10px;border-radius:8px;")
        self.btn_analyze.clicked.connect(self._analyze); self.btn_analyze.setEnabled(False); lv.addWidget(self.btn_analyze)
        self.prog_bar=QProgressBar(); self.prog_bar.setVisible(False); lv.addWidget(self.prog_bar)
        self.lbl_status=QLabel(""); self.lbl_status.setStyleSheet(f"color:{AXIS_C};font-size:11px;"); self.lbl_status.setWordWrap(True); lv.addWidget(self.lbl_status)
        lv.addStretch(); hbox.addWidget(left)
        self.tabs=QTabWidget()
        # Tab 0: 2D Overlay
        t1=QWidget(); v1b=QVBoxLayout(t1)
        self.overlay_canvas=OverlayCanvas(); v1b.addWidget(self.overlay_canvas)
        r1=QHBoxLayout()
        self.btn_play_overlay=QPushButton("Play"); self.btn_play_overlay.setFixedWidth(80); self.btn_play_overlay.clicked.connect(self._toggle_overlay_play); r1.addWidget(self.btn_play_overlay)
        self.btn_2d_side=QPushButton("Side-by-Side"); self.btn_2d_side.setFixedWidth(120); self.btn_2d_side.clicked.connect(self._toggle_2d_side); r1.addWidget(self.btn_2d_side)
        r1.addWidget(QLabel("Frame:")); self.sld_overlay=QSlider(Qt.Horizontal); self.sld_overlay.setRange(0,100); self.sld_overlay.valueChanged.connect(self._update_overlay); r1.addWidget(self.sld_overlay)
        v1b.addLayout(r1); self.tabs.addTab(t1,"2D Overlay")
        # Tab 1: 3D View
        t2=QWidget(); v2b=QVBoxLayout(t2)
        self.skel3d=Skeleton3DCanvas(); self.skel3d.frame_changed.connect(self._on_3d_fc); v2b.addWidget(self.skel3d)
        r2a=QHBoxLayout()
        for lbl,fn in[("Rotate Left",lambda:self.skel3d.rotate(-15)),("Play",self.skel3d.play),("Pause",self.skel3d.pause),("Rotate Right",lambda:self.skel3d.rotate(15))]:
            b=QPushButton(lbl); b.clicked.connect(fn); r2a.addWidget(b)
        self.btn_3d_mode=QPushButton("Side-by-Side"); self.btn_3d_mode.clicked.connect(self._toggle_3d); r2a.addWidget(self.btn_3d_mode); v2b.addLayout(r2a)
        r2b=QHBoxLayout(); r2b.addWidget(QLabel("Frame:")); self.sld_3d=QSlider(Qt.Horizontal); self.sld_3d.setRange(0,100); self.sld_3d.valueChanged.connect(self._on_3d_slider); r2b.addWidget(self.sld_3d); v2b.addLayout(r2b); self.tabs.addTab(t2,"3D View")
        # Tab 2: Joint Analysis
        t3=QWidget(); v3b=QVBoxLayout(t3)
        self.joint_canvas=JointAnalysisCanvas()
        sa3=QScrollArea(); sa3.setWidgetResizable(True); sa3.setWidget(self.joint_canvas)
        v3b.addWidget(sa3); self.tabs.addTab(t3,"Joint Analysis")
        # Tab 3: Swing Analysis
        t4=QWidget(); v4b=QVBoxLayout(t4); v4b.setContentsMargins(0,0,0,0)
        ctrl_row=QHBoxLayout()
        self.btn_toggle_pro_speed=QPushButton("Hide Pro Speed"); self.btn_toggle_pro_speed.setFixedWidth(160); self.btn_toggle_pro_speed.clicked.connect(self._toggle_pro_speed); ctrl_row.addWidget(self.btn_toggle_pro_speed); ctrl_row.addStretch(); v4b.addLayout(ctrl_row)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{background:transparent;}QScrollBar:vertical{background:#16213e;width:12px;}QScrollBar::handle:vertical{background:#0f3460;border-radius:6px;}")
        inner=QWidget(); inner.setStyleSheet(f"background:{DARK};"); inner_layout=QVBoxLayout(inner); inner_layout.setContentsMargins(4,4,4,4)
        self.swing_canvas=SwingAnalysisCanvas(); inner_layout.addWidget(self.swing_canvas)
        traj_sep=QLabel("Ball Trajectory"); traj_sep.setAlignment(Qt.AlignCenter)
        traj_sep.setStyleSheet(f"color:{BLUE};font-size:13px;font-weight:bold;padding:8px;"); inner_layout.addWidget(traj_sep)
        self.traj_canvas=BallTrajectoryCanvas(); inner_layout.addWidget(self.traj_canvas)
        rot_row=QHBoxLayout(); rot_row.addWidget(QLabel("  Rotate:"))
        self.sld_traj_rot=QSlider(Qt.Horizontal); self.sld_traj_rot.setRange(-180,180); self.sld_traj_rot.setValue(-55); self.sld_traj_rot.valueChanged.connect(self._on_traj_rot); rot_row.addWidget(self.sld_traj_rot)
        rot_row.addWidget(QLabel("(L/R)")); rot_row.addSpacing(20); rot_row.addWidget(QLabel("  Wind:"))
        self.sld_traj_wind=QSlider(Qt.Horizontal); self.sld_traj_wind.setRange(-15,15); self.sld_traj_wind.setValue(0); self.sld_traj_wind.setFixedWidth(160); self.sld_traj_wind.valueChanged.connect(self._on_wind_change); rot_row.addWidget(self.sld_traj_wind)
        self.lbl_wind=QLabel("None"); self.lbl_wind.setFixedWidth(120); self.lbl_wind.setStyleSheet(f"color:#88ddff;font-size:11px;"); rot_row.addWidget(self.lbl_wind)
        btn_wr=QPushButton("Reset Wind"); btn_wr.setFixedWidth(100); btn_wr.clicked.connect(self._reset_wind); rot_row.addWidget(btn_wr)
        inner_layout.addLayout(rot_row); inner_layout.addStretch(); scroll.setWidget(inner); v4b.addWidget(scroll); self.tabs.addTab(t4,"Swing Analysis")
        # Tab 4: Video Overlay
        self.video_overlay_tab=VideoOverlayWidget(); self.tabs.addTab(self.video_overlay_tab,"Video Overlay")
        # Tab 5: AI Feedback
        t5=QWidget(); v5=QVBoxLayout(t5)
        ai_header=QHBoxLayout()
        ai_lbl=QLabel("AI Coaching Feedback"); ai_lbl.setStyleSheet(f"font-size:14px;font-weight:bold;color:{BLUE};"); ai_header.addWidget(ai_lbl)
        self.btn_ai=QPushButton("Generate AI Feedback"); self.btn_ai.setFixedWidth(200); self.btn_ai.clicked.connect(self._request_ai_feedback); self.btn_ai.setEnabled(False); ai_header.addWidget(self.btn_ai); ai_header.addStretch()
        v5.addLayout(ai_header)
        self.lbl_ai_status=QLabel("Analyse a stroke first, then click Generate AI Feedback.")
        self.lbl_ai_status.setStyleSheet(f"color:{AXIS_C};font-size:11px;"); v5.addWidget(self.lbl_ai_status)
        self.ai_feedback_box=QTextEdit(); self.ai_feedback_box.setReadOnly(True)
        self.ai_feedback_box.setStyleSheet(f"background:#0d1220;color:white;border:1px solid {ACCENT};font-size:13px;")
        self.ai_feedback_box.setPlaceholderText("AI coaching feedback will appear here...\n\nRequires ANTHROPIC_API_KEY environment variable.")
        v5.addWidget(self.ai_feedback_box); self.tabs.addTab(t5,"AI Feedback")
        hbox.addWidget(self.tabs)

    def _sel_speed(self,combo): return SPEED_OPTIONS[combo.currentIndex()][1]
    def _toggle_3d(self): s=self.skel3d.toggle_side(); self.btn_3d_mode.setText("Overlay Mode" if s else "Side-by-Side")
    def _on_3d_fc(self,f): self.sld_3d.blockSignals(True); self.sld_3d.setValue(f); self.sld_3d.blockSignals(False)
    def _on_3d_slider(self,v): self.skel3d.set_frame(v)
    def _toggle_2d_side(self):
        s=self.overlay_canvas.toggle_side(); self.btn_2d_side.setText("Overlay Mode" if s else "Side-by-Side")
        self._update_overlay(self.sld_overlay.value())
    def _toggle_pro_speed(self):
        cur=self.swing_canvas._show_pro; self.swing_canvas.set_show_pro(not cur)
        self.btn_toggle_pro_speed.setText("Show Pro Speed" if cur else "Hide Pro Speed")
        if self.results: self._replot_swing()
    def _on_traj_rot(self,v): self.traj_canvas.set_azim(v)
    def _on_wind_change(self,v):
        self.traj_canvas.set_wind(float(v))
        if v==0: self.lbl_wind.setText("None")
        elif v>0: self.lbl_wind.setText(f"Tailwind +{v} m/s")
        else: self.lbl_wind.setText(f"Headwind {v} m/s")
    def _reset_wind(self): self.sld_traj_wind.setValue(0); self.lbl_wind.setText("None")
    def _load_pro_cache(self):
        for stroke in ('forehand','backhand','serve'):
            try: self.pro_cache[stroke]=load_reference(self.base_prof_dir,stroke)
            except: pass
    def _pick_video(self):
        path,_=QFileDialog.getOpenFileName(self,"Select Your Tennis Video","","Video Files (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if path:
            self.video_path=path; self._vid_stopped_frame=0
            self.lbl_vid.setText(os.path.basename(path)); self.btn_analyze.setEnabled(True)
            self.lbl_status.setText("Video ready. Press Analyze Stroke.")
            self.pro_vid_path=None; self.lbl_pro.setText("Or use folder: data/professional/<stroke>/")
    def _pick_pro(self):
        path,_=QFileDialog.getOpenFileName(self,"Select Pro Video","","Video Files (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if path: self.pro_vid_path=path; self.lbl_pro.setText(os.path.basename(path))
    def _analyze(self):
        if not self.video_path: return
        stroke=self.combo_stroke.currentText().lower()
        u_sf=self._sel_speed(self.combo_user_speed); p_sf=self._sel_speed(self.combo_pro_speed)
        if self.pro_vid_path:
            from processing.pose_extractor import PoseExtractor
            ext=PoseExtractor(); pro_data=ext.extract_from_video(self.pro_vid_path)
            self.current_pro_label=f"Comparing with: {parse_pro_name(self.pro_vid_path)} (manual)"
        else:
            try:
                pro_data=load_or_build_reference(self.base_prof_dir,stroke,target_len=120,verbose=True)
                self.pro_cache[stroke]=pro_data
                n=pro_data.get("num_videos","?")
                self.current_pro_label=f"Finding best match from {n} pro clips..."
                self.lbl_status.setText(self.current_pro_label)
            except:
                pro_data=_make_synthetic_pro(stroke); self.current_pro_label="Pro: synthetic"
                self.lbl_status.setText("No pro folder – using synthetic reference.")
        self.btn_analyze.setEnabled(False); self.prog_bar.setVisible(True); self.prog_bar.setValue(0)
        self._proc_thread=ProcessingThread(self.video_path,stroke,pro_data,
                                           lbl=self.current_pro_label,usr_sf=u_sf,pro_sf=p_sf)
        self._proc_thread.progress.connect(self._on_progress); self._proc_thread.finished.connect(self._on_done); self._proc_thread.error.connect(self._on_error); self._proc_thread.start()
    def _on_progress(self,msg,val): self.lbl_status.setText(msg); self.prog_bar.setValue(val)
    def _on_done(self,res):
        self.results=res; self.btn_analyze.setEnabled(True); self.prog_bar.setVisible(False)
        pro_name=parse_pro_name(res.get('medoid_video',''))
        self.pro_name=pro_name
        sf_u=res['speed_factor_user']
        self.lbl_status.setText(f"Done. Comparing with: {pro_name}  |  speed: {sf_u:.3f}x")
        uw=normalise_world_seq(res['user_data']['landmarks_3d'])
        pw=normalise_world_seq(res['pro_data']['landmarks_3d'])
        # Raw sequences for swing analysis (phase frame indices are raw frame numbers)
        self.user_world=uw; self.pro_world=pw; n=len(uw)
        # Trim to actual stroke window then linearly resample to 101 frames.
        # Same pure-linear approach as video overlay — guarantees speed parity.
        stroke_type=res['stroke_type']
        u_phases=res.get('phases_user') or {}
        p_phases=res.get('phases_pro')  or {}
        self.user_world_r=trim_resample_linear(uw, u_phases, stroke_type, n_out=101)
        self.pro_world_r =trim_resample_linear(pw, p_phases, stroke_type, n_out=101)
        # Skeleton sliders: 0-100 stroke-progress %. Explicit reset to 0.
        self.sld_overlay.setRange(0,100)
        self.sld_overlay.blockSignals(True); self.sld_overlay.setValue(0); self.sld_overlay.blockSignals(False)
        self.sld_3d.setRange(0,100)
        self.sld_3d.blockSignals(True); self.sld_3d.setValue(0); self.sld_3d.blockSignals(False)
        # Propagate pro_name to all display canvases
        self.overlay_canvas.set_pro_name(pro_name)
        self.skel3d.set_sequences(self.user_world_r, self.pro_world_r,
                                   fps=res['user_data']['fps'], pro_name=pro_name)
        self._update_overlay(0)
        self.joint_canvas.plot(res['errors'], pro_name=pro_name)
        self._replot_swing()
        # Show rule-based coaching immediately -- no API needed
        coaching_md=generate_detailed_coaching(res,res['stroke_type'],pro_name)
        self.ai_feedback_box.setHtml(coaching_md)
        self.lbl_ai_status.setText(
            f"Coaching analysis vs {pro_name} ready. "
            "Click 'Enhance with AI' for personalised Claude coaching (needs ANTHROPIC_API_KEY).")
        self.btn_ai.setEnabled(True); self.btn_ai.setText("Enhance with AI")
        pro_vid=self.pro_vid_path or res.get('medoid_video')
        _u_phases=res.get('phases_user'); _p_phases=res.get('phases_pro')
        _fps=res['user_data']['fps']
        _ulm=res['user_data'].get('landmarks_2d')
        _plm=res['pro_data'].get('landmarks_2d')
        _st=res['stroke_type']
        QTimer.singleShot(500, lambda: self.video_overlay_tab.set_videos(
            self.video_path, pro_vid, mapping_len=n,
            fps=_fps, user_lm2d=_ulm, pro_lm2d=_plm,
            user_phases=_u_phases, pro_phases=_p_phases, stroke_type=_st))
        self.tabs.setCurrentIndex(0)  # show 2D Overlay first

    def _replot_swing(self):
        res=self.results
        if res is None: return
        try:
            uw=self.user_world   # raw — phase frame indices match phases_user
            fps=float(res['user_data'].get('fps',30))
            sf_u=res['speed_factor_user']; sf_p=res['speed_factor_pro']
            self.swing_canvas.plot(user_lm3d=uw,
                phases_u=res['phases_user'],phases_p=res['phases_pro'],
                phases_u_s=res['phases_user_s'],phases_p_s=res['phases_pro_s'],
                rs_real=res['racket_speed_real'],pro_rs_real=res.get('pro_racket_speed_real'),
                peak_rs=res['peak_speed_real'],rs_cont=res['rs_at_contact'],
                contact_label=res['contact_label'],contact_diff=res['contact_diff'],
                rack_ang=res['racket_angle'],stroke_type=res['stroke_type'],
                sf_u=sf_u,sf_p=sf_p,fps=fps,errors=res['errors'],
                pro_name=self.pro_name)
        except Exception: import traceback; print("SwingCanvas:"); traceback.print_exc()
        try:
            v0_rms=res['rs_at_contact']*SW_METERS; rack_ang=res['racket_angle']
            contact_h=res.get('contact_height_m',0.90); cur_wind=float(self.sld_traj_wind.value())
            measured_spin=res.get('spin_rpm',None)
            measured_launch=res.get('launch_angle',None)
            measured_exit=res.get('exit_kmh',None)
            traj=compute_ball_trajectory_physics(v0_rms,rack_ang,res['stroke_type'],contact_h,cur_wind,
                                                  override_spin_rpm=measured_spin,
                                                  override_launch_deg=measured_launch,
                                                  override_exit_kmh=measured_exit)
            self.traj_canvas.set_trajectory(traj,res['stroke_type'],
                                             v0_rms=v0_rms,rack_ang=rack_ang,contact_h=contact_h,
                                             override_spin=measured_spin,
                                             override_launch=measured_launch,
                                             override_exit=measured_exit)
        except Exception: import traceback; print("TrajCanvas:"); traceback.print_exc()

    def _on_error(self,msg):
        self.btn_analyze.setEnabled(True); self.prog_bar.setVisible(False)
        self.lbl_status.setText("Error"); QMessageBox.critical(self,"Analysis Error",msg)

    def _request_ai_feedback(self):
        if self.results is None: return
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self.lbl_ai_status.setText(
                "Set ANTHROPIC_API_KEY environment variable for AI enhancement. "
                "Rule-based coaching is shown below.")
            return
        self.btn_ai.setEnabled(False); self.btn_ai.setText("Calling AI...")
        self.lbl_ai_status.setText("Sending to Claude... (10-20s)")
        d=dict(self.results,sf_u=self.results['speed_factor_user'])
        self._ai_thread=AIFeedbackThread(d)
        self._ai_thread.finished.connect(self._on_ai_done)
        self._ai_thread.error_sig.connect(self._on_ai_error)
        self._ai_thread.start()

    def _on_ai_done(self,text):
        self.ai_feedback_box.setMarkdown(text)
        pro_name=parse_pro_name(self.results.get('medoid_video',''))
        self.lbl_ai_status.setText(f"Claude AI coaching ready — comparison with {pro_name}")
        self.btn_ai.setEnabled(True); self.btn_ai.setText("Regenerate AI")
        self.tabs.setCurrentIndex(5)

    def _on_ai_error(self,msg):
        pro_name=parse_pro_name(self.results.get('medoid_video','') if self.results else '')
        if self.results:
            coaching_md=generate_detailed_coaching(self.results,self.results['stroke_type'],pro_name)
            self.ai_feedback_box.setHtml(coaching_md + f'<p style="color:#ff5555;margin-top:16px;"><i>AI enhancement failed: {msg}</i></p>')
        self.btn_ai.setEnabled(True); self.btn_ai.setText("Enhance with AI")
        self.lbl_ai_status.setText("AI call failed — rule-based coaching shown above")

    def _toggle_overlay_play(self):
        if self._overlay_playing:
            self._overlay_playing=False; self._overlay_timer.stop(); self.btn_play_overlay.setText("Play")
        else:
            if self.results is None: return
            self._overlay_playing=True
            self._overlay_timer.start(33)   # 33ms/tick — same rate as video overlay
            self.btn_play_overlay.setText("Pause")

    def _advance_overlay(self):
        if self.results is None: return
        nv=(self.sld_overlay.value()+1)%101
        self.sld_overlay.blockSignals(True); self.sld_overlay.setValue(nv)
        self.sld_overlay.blockSignals(False); self._update_overlay(nv)

    def _update_overlay(self,pct):
        if self.results is None or self.user_world_r is None: return
        f=int(np.clip(pct,0,100))
        uw=self.user_world_r; pw=self.pro_world_r
        cx,cy,sc=0.50,0.50,0.12
        def proj(wf):
            lm=np.zeros((wf.shape[0],3),dtype=np.float32)
            lm[:,0]=cx+wf[:,0]*sc; lm[:,1]=cy-wf[:,1]*sc; lm[:,2]=1.0; return lm
        self.overlay_canvas.draw_overlay(proj(uw[f]),proj(pw[f]))


def _make_synthetic_pro(stroke_type,n=90):
    lm3d=np.zeros((n,33,3),dtype=np.float32); t=np.linspace(0,1,n)
    base={0:[0,-0.6,0],11:[-0.18,-0.3,0],12:[0.18,-0.3,0],13:[-0.30,0,0],14:[0.30,0,0],
          15:[-0.40,0.2,0],16:[0.40,0.2,0],23:[-0.12,0.4,0],24:[0.12,0.4,0],
          25:[-0.15,0.8,0],26:[0.15,0.8,0],27:[-0.15,1.2,0],28:[0.15,1.2,0]}
    for f in range(n):
        for idx,pos in base.items(): lm3d[f,idx]=pos
        sw=np.sin(t[f]*np.pi)
        if stroke_type=='forehand': lm3d[f,14,0]+=0.35*sw; lm3d[f,16,0]+=0.50*sw; lm3d[f,16,1]-=0.40*sw
        elif stroke_type=='backhand': lm3d[f,13,0]-=0.35*sw; lm3d[f,15,0]-=0.50*sw; lm3d[f,15,1]-=0.30*sw
        elif stroke_type=='serve': lm3d[f,14,1]-=0.50*sw; lm3d[f,16,1]-=0.65*sw; lm3d[f,16,0]+=0.20*sw
    return dict(landmarks_3d=lm3d,landmarks_2d=np.zeros((n,33,3),dtype=np.float32),
                frame_indices=np.arange(n),fps=30.0,total_frames=n)