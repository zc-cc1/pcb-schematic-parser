"""Render model ownership proposals; never read reference annotations."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def draw_ownership(rgb,output,path):
    image=Image.fromarray(rgb)
    draw=ImageDraw.Draw(image)
    font_path=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(font_path),12) if font_path.exists() else ImageFont.load_default()
    for component in output['components']:
        box=component['bbox_top_left']
        draw.rectangle(box,outline=(140,140,140),width=1)
        draw.text((box[0],max(0,box[1]-14)),component['component_id'],font=font,fill=(60,60,60))
    for candidate in output['pin_candidates']:
        x,y=candidate['point']
        colour=(0,150,70) if candidate['owner_status']=='assigned' else (230,100,0) if candidate['owner_status']=='ambiguous_owner' else (210,35,35)
        draw.ellipse((x-3,y-3,x+3,y+3),outline=colour,width=2)
        if candidate['owner']:
            owner=output['components'][candidate['owner_detection_index']]
            box=owner['bbox_top_left']
            anchor=(min(max(x,box[0]),box[2]),min(max(y,box[1]),box[3]))
            draw.line([(x,y),anchor],fill=colour,width=1)
    image.thumbnail((1500,1500))
    canvas=Image.new('RGB',(max(image.width,680),image.height+62),'white')
    canvas.paste(image,(0,0))
    legend=ImageDraw.Draw(canvas)
    legend.text((8,image.height+4),'Green: assigned | Orange: ambiguous | Red: unresolved',font=font,fill=(60,60,60))
    legend.text((8,image.height+22),'Ownership proposals only; colours do not indicate correctness.',font=font,fill=(60,60,60))
    legend.text((8,image.height+40),'Gray boxes: YOLO. Short links show ownership, not circuit wires.',font=font,fill=(60,60,60))
    canvas.save(path)
