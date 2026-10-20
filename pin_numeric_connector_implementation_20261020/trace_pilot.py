import linecache
from pathlib import Path
import runpy
import sys
def trace(frame,event,arg):
    if event=='line' and frame.f_code.co_name=='recover_numeric_connectors' and frame.f_lineno==158:
        keys=['owner','possible']
        print(frame.f_lineno,linecache.getline(frame.f_code.co_filename,frame.f_lineno).strip(),str({k:v for k,v in frame.f_locals.items() if k in keys})[:2200])
    return trace
sys.settrace(trace)
runpy.run_path(str(Path(__file__).with_name('verify_cached.py')),run_name='__main__')
