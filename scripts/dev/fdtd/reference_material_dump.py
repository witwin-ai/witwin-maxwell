"""Decode native material component arrays from independent VTK XML dumps.

Only the reference writer's inline ASCII/binary and zlib formats are needed;
the validation workflow therefore does not depend on an incompatible VTK wheel.
The vector component at operator index (i,j,k) denotes that component's own
staggered sample, even though the file attaches it to a common plotting grid.
"""
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zlib
import numpy as np


_TYPES={'Float32':'f4','Float64':'f8','UInt32':'u4','UInt64':'u8','Int32':'i4','Int64':'i8'}


def _decode_array(element,root):
    endian='<' if root.get('byte_order','LittleEndian')=='LittleEndian' else '>'
    dtype=np.dtype(endian+_TYPES[element.attrib['type']])
    text=''.join((element.text or '').split())
    if element.get('format','ascii')=='ascii':return np.fromstring(element.text or '',sep=' ',dtype=dtype.newbyteorder('=')).astype(dtype)
    if element.get('format')!='binary':raise ValueError('The independent native material writer must use inline ASCII or binary data.')
    header=np.dtype(endian+_TYPES[root.get('header_type','UInt32')]);size=header.itemsize
    prefix=base64.b64decode(text[:4*((size+2)//3)])
    first=int(np.frombuffer(prefix[:size],dtype=header)[0])
    if root.get('compressor'):
        if root.attrib['compressor']!='vtkZLibDataCompressor':raise ValueError('The independent material dump requires the declared zlib compressor.')
        count=first;header_bytes=(3+count)*size
        header_chars=4*((header_bytes+2)//3)
        segment=text[:header_chars]
        if segment.endswith('='):
            words=np.frombuffer(base64.b64decode(segment)[:header_bytes],dtype=header)
            payload=base64.b64decode(text[header_chars:])
        else:
            raw=base64.b64decode(text)
            words=np.frombuffer(raw[:header_bytes],dtype=header);payload=raw[header_bytes:]
        blocks=[];offset=0
        for number,compressed_size in enumerate(words[3:]):
            compressed_size=int(compressed_size)
            block=zlib.decompress(payload[offset:offset+compressed_size]);offset+=compressed_size
            expected=int(words[2] or words[1]) if number==count-1 else int(words[1])
            if len(block)!=expected:raise ValueError('Independent material block length disagrees with its compression header.')
            blocks.append(block)
        raw=b''.join(blocks)
    else:
        raw=base64.b64decode(text)
        if len(raw)>=size+first:raw=raw[size:size+first]
        else:raw=base64.b64decode(text[4*((size+2)//3):])
        if len(raw)!=first:raise ValueError('Independent material byte count disagrees with its header.')
    return np.frombuffer(raw,dtype=dtype).copy()


def read_native_material_dump(path):
    """Return [component,i,j,k] arrays with exact reference index association."""
    root=ET.parse(path).getroot();piece=root.find('.//Piece')
    if piece is None:raise ValueError('The independent material dump must contain one structured grid piece.')
    extent=list(map(int,piece.attrib['Extent'].split()))
    dimensions=tuple(extent[2*i+1]-extent[2*i]+1 for i in range(3))
    result={'dimensions':dimensions,'extent':tuple(extent),'point_data':{}}
    for element in piece.findall('./PointData/DataArray'):
        components=int(element.get('NumberOfComponents','1'))
        values=_decode_array(element,root)
        values=values.reshape(dimensions[2],dimensions[1],dimensions[0],components).transpose(3,2,1,0).copy()
        result['point_data'][element.attrib['Name']]=values
    coordinates=piece.findall('./Coordinates/DataArray')
    if coordinates:result['axes']=tuple(_decode_array(element,root) for element in coordinates)
    points=piece.find('./Points/DataArray')
    if points is not None:
        result['world_points']=_decode_array(points,root).reshape(dimensions[2],dimensions[1],dimensions[0],3).transpose(2,1,0,3).copy()
    return result


def save_native_material_arrays(source,destination):
    data=read_native_material_dump(source)
    arrays=dict(data['point_data']);arrays['dimensions']=np.asarray(data['dimensions'])
    if 'axes' in data:arrays.update(zip(('x_nodes','y_nodes','z_nodes'),data['axes']))
    if 'world_points' in data:arrays['world_points']=data['world_points']
    np.savez_compressed(destination,**arrays)
    return dict(dimensions=data['dimensions'],components={name:list(values.shape) for name,values in data['point_data'].items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();print(json.dumps(save_native_material_arrays(args.source,args.destination)))
