"""Independent VTK material arrays preserve operator component/index layout."""
import base64
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest

from scripts.dev.fdtd.reference_material_dump import read_native_material_dump, save_native_material_arrays


def encoded(values, *, header_type, compressed, split_header, endian):
    raw=values.tobytes()
    integer=np.dtype(endian+('u4' if header_type=='UInt32' else 'u8'))
    if compressed:
        # Deliberately unequal last block exercises the writer's blocked framing.
        block_size=40
        blocks=[zlib.compress(raw[start:start+block_size]) for start in range(0,len(raw),block_size)]
        header=np.asarray([len(blocks),block_size,len(raw)%block_size,*map(len,blocks)],dtype=integer).tobytes()
        payload=b''.join(blocks)
    else:
        header=np.asarray([len(raw)],dtype=integer).tobytes();payload=raw
    if split_header:return base64.b64encode(header).decode()+base64.b64encode(payload).decode()
    return base64.b64encode(header+payload).decode()


@pytest.mark.parametrize('header_type',('UInt32','UInt64'))
@pytest.mark.parametrize('compressed',(False,True))
@pytest.mark.parametrize('split_header',(False,True))
@pytest.mark.parametrize('endian',('<','>'))
def test_actual_component_staggers_keep_vtk_fortran_association(tmp_path,header_type,compressed,split_header,endian):
    dimensions=(3,4,2)
    i,j,k=np.indices(dimensions)
    components=np.stack([1000*c+100*i+10*j+k for c in range(3)]).astype(endian+'f8')
    writer_order=components.transpose(3,2,1,0).copy().reshape(-1)
    attributes={'type':'StructuredGrid','byte_order':'LittleEndian' if endian=='<' else 'BigEndian','header_type':header_type}
    if compressed:attributes['compressor']='vtkZLibDataCompressor'
    root=ET.Element('VTKFile',attributes)
    grid=ET.SubElement(root,'StructuredGrid',WholeExtent='0 2 0 3 0 1')
    piece=ET.SubElement(grid,'Piece',Extent='0 2 0 3 0 1')
    data=ET.SubElement(piece,'PointData')
    array=ET.SubElement(data,'DataArray',Name='epsilon',type='Float64',NumberOfComponents='3',format='binary')
    array.text=encoded(writer_order,header_type=header_type,compressed=compressed,split_header=split_header,endian=endian)
    points=np.stack((i+.1*j,j-.2*k,k+.3*i),-1)
    xyz=ET.SubElement(ET.SubElement(piece,'Points'),'DataArray',type='Float64',NumberOfComponents='3',format='ascii')
    xyz.text=' '.join(map(str,points.transpose(2,1,0,3).reshape(-1)))
    path=tmp_path/'material_dump.vts';ET.ElementTree(root).write(path)
    actual=read_native_material_dump(path)
    assert actual['dimensions']==dimensions
    np.testing.assert_array_equal(actual['point_data']['epsilon'],components)
    np.testing.assert_allclose(actual['world_points'],points,rtol=0,atol=0)
    destination=tmp_path/'native.npz'
    save_native_material_arrays(path,destination)
    with np.load(destination) as saved:
        np.testing.assert_array_equal(saved['epsilon'],components)
        np.testing.assert_array_equal(saved['dimensions'],dimensions)


def test_ascii_rectilinear_axes_are_not_staggered_material_coordinates(tmp_path):
    path=tmp_path/'material_dump.vtr'
    path.write_text('''<VTKFile type="RectilinearGrid"><RectilinearGrid><Piece Extent="0 1 0 0 0 0">
      <PointData><DataArray Name="kappa" type="Float32" NumberOfComponents="3" format="ascii">1 2 3 4 5 6</DataArray></PointData>
      <Coordinates><DataArray type="Float64">0 .25</DataArray><DataArray type="Float64">2</DataArray><DataArray type="Float64">3</DataArray></Coordinates>
      </Piece></RectilinearGrid></VTKFile>''')
    actual=read_native_material_dump(path)
    np.testing.assert_array_equal(actual['point_data']['kappa'][:,:,0,0],[[1,4],[2,5],[3,6]])
    np.testing.assert_array_equal(actual['axes'][0],[0,.25])
