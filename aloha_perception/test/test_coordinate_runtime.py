"""Run after sourcing this package's installation; no cameras/GPU/robot needed."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from aloha_tracking_perception.calibration_common import load_camera_calibration
from aloha_tracking_perception.yolo_to_robot import (
    DEFAULT_CALIBRATION, YoloToRobot, default_workspace_parameters,
    transform_detection, workspace_detection,
)
from aloha_tracking_perception.workspace_overlay import WorkspaceOverlay


def test_bundled_configuration_loads():
    assert Path(DEFAULT_CALIBRATION).is_file()
    optical,T=load_camera_calibration(DEFAULT_CALIBRATION,'follower_left/base_link')
    assert optical=='camera_color_optical_frame'
    assert T.shape==(4,4)
    defaults=default_workspace_parameters()
    assert np.isfinite([defaults['workspace_origin_x'],defaults['workspace_origin_y']]).all()
    assert defaults['workspace_frame']=='table_workspace'


def test_rotation_translation_and_workspace_origin():
    T=np.eye(4)
    T[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    T[:3,3]=[1.,2.,3.]
    incoming=dict(label='truck',confidence=.9,cx_3d=.1,cy_3d=.2,cz_3d=1.,
                  frame_id='optical',stamp=10.,pixel_center=[320,240])
    base=transform_detection(incoming,T,'optical','base',10.1,1.)
    np.testing.assert_allclose([base[k] for k in ('x','y','z')],[.8,2.1,4.])
    base.update(smoothing='ema',filter_alpha=.25)
    workspace=workspace_detection(base,.8,2.1,'table_workspace')
    np.testing.assert_allclose([workspace[k] for k in ('x','y','z')],[0.,0.,4.])
    assert workspace['base_position']['x']==base['x']
    assert workspace['pixel_center']==[320,240]
    assert workspace['camera_frame_id']=='optical'


@pytest.mark.parametrize('change',[
    {'frame_id':'wrong'},{'stamp':8.},{'cz_3d':-1.},{'cx_3d':float('nan')},
])
def test_invalid_detection_rejected(change):
    d=dict(label='truck',confidence=.9,cx_3d=.1,cy_3d=.2,cz_3d=1.,frame_id='optical',stamp=10.)
    d.update(change)
    with pytest.raises(ValueError):transform_detection(d,np.eye(4),'optical','base',10.1,1.)


def test_filter_and_loss_clear_workspace_output():
    state=SimpleNamespace(filtered_pub=Mock(),workspace_pub=Mock(),filter_alpha=.25,
        filter_reset_seconds=1.,filtered_xyz=None,filter_stamp=None,
        workspace_origin_x=.5,workspace_origin_y=.02,workspace_frame='table_workspace')
    d=dict(label='truck',confidence=.9,x=.5,y=.02,z=.1,stamp=10.,frame_id='base')
    YoloToRobot.publish_filtered(state,[d])
    result=json.loads(state.workspace_pub.publish.call_args.args[0].data)[0]
    assert [result[k] for k in ('x','y','z')]==[0.,0.,.1]
    YoloToRobot.publish_filtered(state,[])
    assert state.workspace_pub.publish.call_args.args[0].data=='[]'
    assert state.filtered_xyz is None


def test_overlay_matches_source_and_rejects_old_data():
    overlay=WorkspaceOverlay()
    header=SimpleNamespace(frame_id='optical')
    overlay.remember(10.,np.zeros((480,640,3),dtype=np.uint8),header,[(320,240)])
    d=dict(x=0.,y=0.,z=.1,stamp=10.,frame_id='table_workspace',units='m',
           smoothing='ema',camera_frame_id='optical',pixel_center=[320,240])
    assert overlay.render([d],10.1) is not None
    overlay.remember(10.,np.zeros((480,640,3),dtype=np.uint8),header,[(320,240)])
    assert overlay.render([d],12.) is None
