"""Download documented public CAD/PDF sources and keep immutable provenance."""
import argparse
import hashlib
import json
import re
import shlex
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_sexpression(text):
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|[()]|[^\s()]+', text)
    stack, root = [], []
    current = root
    for token in tokens:
        if token == "(":
            child=[];current.append(child);stack.append(current);current=child
        elif token == ")":
            if not stack: raise ValueError("Unbalanced S-expression")
            current=stack.pop()
        else:
            current.append(json.loads(token, strict=False) if token.startswith('"') else token)
    if stack or len(root)!=1: raise ValueError("Unbalanced S-expression")
    return root[0]


def entries(node, key):
    return [v for v in node if isinstance(v,list) and v and v[0]==key]


def cad_inventory(path):
    try:
        text=path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("Binary CAD file retained; requires a CAD converter before native annotation extraction") from None
    components=[]
    if text.lstrip().startswith("<?xml") or "<eagle" in text[:300]:
        tree=ET.fromstring(text)
        for part in tree.findall(".//schematic/parts/part"):
            components.append({"reference":part.get("name"),"value":part.get("value",""),
                               "library_id":part.get("library","")+":"+part.get("deviceset","")})
        nets=[{"name":net.get("name"),"pin_references":[dict(v.attrib) for v in net.findall(".//pinref")]} for net in tree.findall(".//schematic/sheets/sheet/nets/net")]
        kind="eagle_xml"
        sheet_count=len(tree.findall(".//schematic/sheets/sheet"))
    elif text.startswith("EESchema Schematic File"):
        for block in re.findall(r"\$Comp\s*\n(.*?)\$EndComp",text,re.S):
            lines=block.splitlines()
            library=next(shlex.split(v) for v in lines if v.startswith("L "))
            value=next((shlex.split(v)[2] for v in lines if v.startswith("F 1 ")),"")
            point=next((shlex.split(v)[1:] for v in lines if v.startswith("P ")),[])
            components.append({"reference":library[2],"value":value,"library_id":library[1],"position_mil":[float(v) for v in point]})
        nets=[];kind="kicad_legacy";sheet_count=1
    elif text.lstrip().startswith("(kicad_sch"):
        tree=read_sexpression(text)
        for symbol in entries(tree,"symbol"):
            props={p[1]:p[2] for p in entries(symbol,"property")}
            libs=entries(symbol,"lib_id")
            positions=entries(symbol,"at")
            components.append({"reference":props.get("Reference"),"value":props.get("Value",""),
                               "library_id":libs[0][1] if libs else "","position_mm":positions[0][1:3] if positions else [],
                               "properties":props})
        nets=[];kind="kicad_sexpression";sheet_count=1
    else:
        raise ValueError("Unsupported CAD encoding; source retained for review")
    targets=[]
    for obj in components:
        token=(obj["library_id"]+" "+obj["value"]).upper()
        hints=[]
        if re.search(r"CRYSTAL|XTAL|OSCILLATOR|RESONATOR|QUARTZ",token): hints.append("clock_symbol_candidate")
        if re.search(r"MOSFET|[NP]MOS|BSS\d|IRL\w|WPM\d|SI230\w|2N700\w|DMG\d",token): hints.append("mosfet_candidate")
        if hints: targets.append({**obj,"hints":hints,"verified_class":None})
    return {"format":kind,"component_instance_count":len(components),"sheet_count":sheet_count,"components":components,
            "nets_from_native_xml":nets,"target_candidates":targets,
            "status":"native_inventory_not_pixel_annotations_not_competition_Golden"}


def download(source, item, folder, maximum):
    relative=Path(item["path"])
    if relative.is_absolute() or ".." in relative.parts: raise ValueError("Unsafe source path")
    path=folder/"raw"/source["id"]/relative
    url=item.get("url") or "https://raw.githubusercontent.com/"+source["repository"]+"/"+source["ref"]+"/"+urllib.parse.quote(item["path"],safe="/")
    record={"source_id":source["id"],"group":source["group"],"role":item["role"],"repository_path":item["path"],
            "url":url,"local_path":str(path),"optional":item.get("optional",False)}
    previous=path.with_name(path.name+".provenance.json")
    try:
        if path.exists() and previous.exists():
            old=json.loads(previous.read_text(encoding="utf-8"))
            if digest(path)!=old["sha256"]: raise ValueError("Cached source differs from recorded hash")
            return old
        request=urllib.request.Request(url,headers={"User-Agent":"PCB-schematic-research-collection/0.1"})
        with urllib.request.urlopen(request,timeout=30) as response:
            expected=response.headers.get("Content-Length")
            if expected and int(expected)>maximum: raise ValueError("Source exceeds configured size limit")
            data=response.read(maximum+1)
            if len(data)>maximum: raise ValueError("Source exceeds configured size limit")
            record.update(content_type=response.headers.get("Content-Type"),etag=response.headers.get("ETag"))
        if item["role"]=="schematic_pdf" and not data.startswith(b"%PDF-"): raise ValueError("Expected PDF signature")
        if path.exists() and path.read_bytes()!=data: raise ValueError("Refusing to replace a previously collected raw source")
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        record.update(status="downloaded",bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                      downloaded_at_utc=datetime.now(timezone.utc).isoformat(),ref=source["ref"],commit=None,
                      revision_status="branch_ref_with_content_sha256_commit_lookup_unavailable")
        write_json(previous,record)
    except Exception as exc:
        record.update(status="unavailable",error=f"{type(exc).__name__}: {str(exc)[:400]}")
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"configs/external_sources.json")
    parser.add_argument("--source-id",action="append")
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding="utf-8"))
    sources=[s for s in cfg["sources"] if not args.source_id or s["id"] in args.source_id]
    folder=ROOT/cfg["output_directory"];folder.mkdir(parents=True,exist_ok=True)
    all_files=[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(download,source,item,folder,cfg["maximum_file_bytes"]) for source in sources for item in source["files"]]
        for future in as_completed(futures):
            row=future.result();all_files.append(row)
            print(json.dumps({"source":row["source_id"],"file":row["repository_path"],"status":row["status"]},ensure_ascii=False),flush=True)
    all_files.sort(key=lambda r:(r["source_id"],r["repository_path"]))
    cases=[];inventories=[];issues=[]
    for source in sources:
        source_files=[r for r in all_files if r["source_id"]==source["id"] and r["status"]=="downloaded"]
        for record in source_files:
            path=Path(record["local_path"])
            if record["role"]=="schematic_cad":
                try:
                    inventory=cad_inventory(path)
                    write_json(folder/"native_inventory"/source["id"]/(path.stem+".json"),{**inventory,"source":record})
                    inventories.append({"source_id":source["id"],"cad_sha256":record["sha256"],
                                        "format":inventory["format"],"component_instance_count":inventory["component_instance_count"],
                                        "sheet_count":inventory["sheet_count"],"target_candidates":inventory["target_candidates"]})
                except Exception as exc: issues.append({"source_id":source["id"],"path":str(path),"issue":"cad_inventory_failed","error":f"{type(exc).__name__}: {str(exc)[:400]}"})
            if record["role"]!="schematic_pdf": continue
            with pdfplumber.open(path) as pdf:
                for number,page in enumerate(pdf.pages,1):
                    case_id=f"ext_{source['id']}_p{number:02d}"
                    image_path=folder/"images"/(case_id+".png");image_path.parent.mkdir(parents=True,exist_ok=True)
                    page.to_image(resolution=cfg["render_dpi"]).save(image_path)
                    with __import__('PIL.Image',fromlist=['Image']).open(image_path) as im: width,height=im.size
                    text=page.extract_text() or ""
                    text_path=folder/"page_text"/(case_id+".txt");text_path.parent.mkdir(parents=True,exist_ok=True);text_path.write_text(text,encoding="utf-8")
                    cases.append({"case_id":case_id,"source_id":source["id"],"source_group":source["group"],
                                  "source_repository":"https://github.com/"+source["repository"],"source_file_url":record["url"],
                                  "pdf_sha256":record["sha256"],"page":number,"pdf_page_points":[page.width,page.height],
                                  "render_dpi":cfg["render_dpi"],"image":str(image_path),"image_sha256":digest(image_path),
                                  "width":width,"height":height,"native_cad_paths":[r["local_path"] for r in source_files if r["role"]=="schematic_cad"],
                                  "hardware_license":source["hardware_license"],"license_evidence":source["evidence"],
                                  "annotation_status":"unannotated_native_CAD_available","training_eligible":False,
                                  "bonus_eligible":None,"bonus_status":"pending_official_rules_full_correct_annotation_and_deduplication"})
    (folder/"manifest.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in cases)+"\n",encoding="utf-8")
    write_json(folder/"source_files.json",all_files)
    write_json(folder/"native_summary.json",inventories)
    audit={"created_at_utc":datetime.now(timezone.utc).isoformat(),"sources_requested":len(sources),
           "sources_with_native_cad":len({r["source_id"] for r in all_files if r["role"]=="schematic_cad" and r["status"]=="downloaded"}),
           "sources_with_parsed_native_cad":len({r["source_id"] for r in inventories}),"rendered_schematic_pages":len(cases),
           "rendered_source_groups":len({r["source_group"] for r in cases}),"downloaded_files":sum(r["status"]=="downloaded" for r in all_files),
           "downloaded_bytes":sum(r.get("bytes",0) for r in all_files),"unavailable_files":[r for r in all_files if r["status"]!="downloaded"],
           "native_component_instances":sum(r["component_instance_count"] for r in inventories),"issues":issues,
           "sources":sources,"training_eligible_pages":0,"official_bonus_score":None,"status":"collected_candidates_pending_deduplication_annotation_and_review"}
    write_json(folder/"audit.json",audit)
    print(json.dumps({k:audit[k] for k in ("sources_with_native_cad","rendered_schematic_pages","rendered_source_groups","downloaded_files","native_component_instances","training_eligible_pages")},ensure_ascii=False),flush=True)


if __name__=="__main__": main()
