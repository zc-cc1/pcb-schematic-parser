"""Move existing free wire endpoints to unique raster green/red contacts.

This finalizer consumes prediction geometry and the current RGB image only.
It preserves edge identities, all electrical membership, components and pins.
No reference annotations, filenames, page identities or score tolerances enter
the algorithm. Every proposal has a continuous thin green arm and an adjoining
thin red lead; uncertain, crossed, disconnected or budget-exceeding work abstains.
"""
from copy import deepcopy
import hashlib
import math
import numpy as np

DEFAULT={'enabled':False,'maximum_changes':512,'search_radius':24}

def _mask(rgb):
 r,g,b=[rgb[:,:,i].astype(np.int16) for i in range(3)]
 return ((g>r+20)&(g>b+20)&(g>60),(r>g+35)&(r>b+35)&(g<150)&(b<150),
         (np.maximum(r,g)>b+20)&(np.maximum(r,g)>40)&(np.minimum(r,g)<200))

def _groups(positions):
 if not len(positions):return []
 return np.split(positions,np.flatnonzero(np.diff(positions)>1)+1)

def _nearest_pixel(value):
 # Unlike bankers' round, translating half-pixel coordinates by one pixel
 # must visit every successive raster column/row exactly once.
 return math.floor(float(value)+.5)

def _point_segment_distance(point,a,b):
 vector=b-a;den=float(vector@vector)
 if den<=1e-12:return math.dist(point,a)
 return math.dist(point,a+np.clip(float((point-a)@vector)/den,0.,1.)*vector)

def _segment_distance(a,b,c,d):
 def cross(u,v):return float(u[0]*v[1]-u[1]*v[0])
 u,v=b-a,d-c;den=cross(u,v)
 if abs(den)>1e-12:
  t,s=cross(c-a,v)/den,cross(c-a,u)/den
  if 0<=t<=1 and 0<=s<=1:return 0.
 return min(_point_segment_distance(a,c,d),_point_segment_distance(b,c,d),
            _point_segment_distance(c,a,b),_point_segment_distance(d,a,b))

def _contact(green,red,color,a,b,radius):
 h,w=green.shape;vector=b-a;axis=int(abs(vector[1])>abs(vector[0]));length=abs(vector[axis])
 if length<28 or abs(vector[1-axis])>max(2.,length*.04):return None
 direction=1 if vector[axis]>0 else -1
 # Use the inner part of the *existing* edge to measure the thin wire axis.
 # Never infer an endpoint by extending a component bounding box or pin.
 levels=[];widths=[];observed=[]
 for distance in range(0,min(int(length)-4,radius+16)):
  p=a.copy();p[axis]+=direction*distance
  x,y=map(_nearest_pixel,[p[0],h-p[1]])
  if not(8<=x<w-8 and 8<=y<h-8):return None
  strip=green[y-6:y+7,x] if axis==0 else green[y,x-6:x+7]
  runs=_groups(np.flatnonzero(strip))
  if not runs:continue
  if len(runs)!=1 or not 1<=len(runs[0])<=5:return None
  centre=float(np.mean(runs[0]));widths.append(len(runs[0]))
  levels.append(h-(y-6+centre) if axis==0 else x-6+centre);observed.append(distance)
 if len(levels)<12 or np.ptp(levels)>1:return None
 if max(map(len,_groups(np.array(observed))))<12:return None
 level=float(np.median(levels));width=float(np.median(widths))
 if abs(level-a[1-axis])>4:return None
 def position(distance):
  p=a.copy();p[axis]+=direction*distance;p[1-axis]=level
  return p
 def strip(mask,distance,half=1):
  p=position(distance);x,y=map(_nearest_pixel,[p[0],h-p[1]])
  if not(8<=x<w-8 and 8<=y<h-8):return None
  return mask[y-half:y+half+1,x] if axis==0 else mask[y,x-half:x+half+1]
 distances=list(range(-radius,radius+13))
 gs={d:bool(strip(green,d).any()) if strip(green,d) is not None else False for d in distances}
 rs={d:bool(strip(red,d).any()) if strip(red,d) is not None else False for d in distances}
 cs={d:bool(strip(color,d).any()) if strip(color,d) is not None else False for d in distances}
 candidates=[]
 for distance in range(-radius+6,radius+1):
  if not all(gs.get(d,False) for d in range(distance,distance+12)):continue
  # JPEG/resampling may create one or two dark mixed-color pixels between
  # green and red. Accept those only if every intermediate pixel has chromatic
  # ink; even a single white pixel is a real disconnection and must abstain.
  gaps=[gap for gap in range(3)
        if all(cs.get(d,False) and not gs.get(d,False) for d in range(distance-gap,distance))
        and all(rs.get(d,False) and not gs.get(d,False) for d in range(distance-gap-4,distance-gap))]
  if not gaps:continue
  gap=min(gaps)
  # Thick red junctions and perpendicular green crossings do not prove a pin
  # contact. Their transverse extent breaks the thin-lead/arm contract.
  valid=True
  for mask,offsets in ((red,range(distance-gap-4,distance-gap)),(green,range(distance,distance+12))):
   for d in offsets:
    sample=strip(mask,d,6);runs=_groups(np.flatnonzero(sample)) if sample is not None else []
    if len(runs)!=1 or len(runs[0])>width+2:valid=False;break
   if not valid:break
  if valid:candidates.append(position(distance))
 if len(candidates)!=1:return None
 proposed=candidates[0]
 proposed[axis]=(_nearest_pixel(proposed[0]) if axis==0
                 else h-_nearest_pixel(h-proposed[1]))
 if math.dist(a,proposed)>radius:return None
 return proposed,{'axis':axis,'green_axis':level,'green_width':width,
                  'continuous_green_samples':12,'contiguous_red_samples':4}

def refine_wire_contacts(target,rgb,image_sha256,*,image_binding,cfg=None):
 config={**DEFAULT,**(cfg or {})}
 if set(config)!=set(DEFAULT) or type(config['enabled']) is not bool:raise ValueError('invalid_precise_wire_contact_setting')
 if type(config['maximum_changes']) is not int or not 0<=config['maximum_changes']<=2048:raise ValueError('invalid_precise_wire_contact_budget')
 if type(config['search_radius']) is not int or not 8<=config['search_radius']<=32:raise ValueError('invalid_precise_wire_contact_search_radius')
 if not config['enabled']:return deepcopy(target),{'enabled':False,'status':'disabled','changes':[]}
 rgb=np.asarray(rgb)
 if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:raise ValueError('precise_wire_contact_invalid_rgb_image')
 h,w=rgb.shape[:2]
 if (not isinstance(image_sha256,str) or not image_sha256 or image_binding.get('image_sha256')!=image_sha256
     or image_binding.get('size')!=[w,h] or image_binding.get('coordinate_origin')!='top_left'
     or image_binding.get('prediction_only') is not True
     or image_binding.get('rgb_sha256')!=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()):
  raise ValueError('precise_wire_contact_evidence_differs_from_image')
 green,red,color=_mask(rgb);out=deepcopy(target);events=[];segments=[]
 for nid,net in target.get('nets',{}).items():
  for eid,edge in net.get('edges',{}).items():
   if not isinstance(edge,list) or len(edge)!=2:continue
   try:ps=np.array([[p['x'],p['y']] for p in edge],float)
   except (ValueError,KeyError,TypeError):continue
   if np.isfinite(ps).all():segments.append((nid,eid,ps))
 endpoints=[p for _,_,ps in segments for p in ps]
 for nid,eid,ps in segments:
  for index,point in enumerate(ps):
   if sum(math.dist(point,other)<=2 for other in endpoints)!=1:continue
   proposal=_contact(green,red,color,point,ps[1-index],config['search_radius'])
   if proposal is None:continue
   corrected,proof=proposal
   if math.dist(point,corrected)<.5:continue
   # Endpoint counting alone misses a T whose contact lies inside another
   # retained segment. Reject source/proposal contacts and any swept movement
   # crossing, including across logical nets. Also reject a new intersection
   # produced by lateral recentering of the complete edge.
   others=[other for nn,ee,other in segments if (nn,ee)!=(nid,eid)]
   if any(_segment_distance(point,corrected,*other)<=2 for other in others):continue
   other_end=ps[1-index]
   if any(_segment_distance(corrected,other_end,*other)<=2
          and _segment_distance(point,other_end,*other)>2 for other in others):continue
   out['nets'][nid]['edges'][eid][index].update(x=float(corrected[0]),y=float(corrected[1]))
   events.append({'rule':'observed_green_red_contact','network_id':nid,'edge_id':eid,
                  'endpoint_index':index,'before':point.tolist(),'after':corrected.tolist(),**proof})
 if len(events)>config['maximum_changes']:
  return deepcopy(target),{'enabled':True,'status':'withheld_budget','changes':[],'proposed_changes':len(events)}
 # Validate the complete simultaneous proposal, in both directions. Individual
 # source/proposal checks cannot detect a lost mid-edge T or two independently
 # accepted endpoint moves that together create a contact. Any discrepancy
 # atomically withholds every change, retaining the original contact graph.
 modified={(event['network_id'],event['edge_id']) for event in events}
 final_segments=[]
 for nid,eid,old in segments:
  new=np.array([[p['x'],p['y']] for p in out['nets'][nid]['edges'][eid]],float)
  final_segments.append((nid,eid,old,new))
 unsafe=[]
 for i,(nid,eid,before,after) in enumerate(final_segments):
  for nn,ee,old,new in final_segments[:i]:
   if (nid,eid) not in modified and (nn,ee) not in modified:continue
   contact_before=_segment_distance(*before,*old)<=2
   contact_after=_segment_distance(*after,*new)<=2
   if contact_before!=contact_after:
    unsafe.append({'edges':[[nn,ee],[nid,eid]],'contact_before':bool(contact_before),
                   'contact_after':bool(contact_after)})
 if unsafe:
  return deepcopy(target),{'enabled':True,'status':'withheld_topology','changes':[],
                          'proposed_changes':len(events),'unsafe_contacts':unsafe}
 # Check the whole non-edge structure, rather than enumerating known fields.
 def structure(doc):
  value=deepcopy(doc)
  for net in value.get('nets',{}).values():
   for edge in net.get('edges',{}).values():
    if not isinstance(edge,list):continue
    for point in edge:
     if isinstance(point,dict):point.pop('x',None);point.pop('y',None)
  return value
 if structure(out)!=structure(target):raise ValueError('precise_wire_contact_modified_structure')
 return out,{'enabled':True,'status':'applied' if events else 'no_effect','changes':events,
             'image_sha256':image_sha256,'rgb_sha256':image_binding['rgb_sha256'],'reference_read':False}
