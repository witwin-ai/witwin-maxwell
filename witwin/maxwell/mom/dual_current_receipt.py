"""Historical streamed dual admission checks for HB and cutoff ownership."""
import math
from dataclasses import asdict
import torch
from .static_projector_snapshot import _validate_bound


def validate_dual_receipts(records,*,rule,column_batch,cell_count,dtype):
    if not isinstance(records,(tuple,list)) or len(records)!=2:
        raise ValueError('Invalid dual DC receipt count')
    nonlinear,current=records
    if not isinstance(nonlinear,dict) or not isinstance(current,dict):
        raise ValueError('Invalid dual DC receipts')
    iterations=nonlinear.get('iterations');checks=nonlinear.get('tangent_checks')
    if (nonlinear.get('linear_solver')!='streamed_dual_checked_gmres'
            or nonlinear.get('streamed_rule')!=asdict(rule)
            or nonlinear.get('streamed_column_batch')!=column_batch
            or nonlinear.get('converged') is not True or nonlinear.get('failure_reason') is not None
            or type(iterations) is not int or not 0<=iterations<=30
            or not isinstance(checks,list) or len(checks)!=iterations+2
            or nonlinear.get('conductivity_derivatives') is not True
            or any(nonlinear.get(k) is not False for k in ('global_response_stored',
                'global_nonlinear_tangent_stored','fast_work_qualified','physical_qualification'))):
        raise ValueError('Stored dual DC nonlinear policy mismatch')
    residual_checks=nonlinear.get('checks')
    if not isinstance(residual_checks,list) or len(residual_checks)!=iterations+1:
        raise ValueError('Invalid dual DC nonlinear residual history')
    for index,entry in enumerate(residual_checks):
        value=entry.get('residual_V_per_m_l2') if isinstance(entry,dict) else None
        if (not isinstance(entry,dict) or type(entry.get('iteration')) is not int
                or entry['iteration']!=index or type(value) not in (int,float)
                or not math.isfinite(value) or value<0):
            raise ValueError('Invalid dual DC nonlinear residual history')
    expected_threshold=1e-12+1e-10*residual_checks[0]['residual_V_per_m_l2']
    if nonlinear.get('threshold_V_per_m_l2')!=expected_threshold:
        raise ValueError('Stored dual DC nonlinear threshold differs from native policy')
    def bound(value,size):
        _validate_bound(value,rule,size)
        if (value['column_batch']!=column_batch
                or value['numerical_singularity_cutoff']!=64*torch.finfo(dtype).eps*size*value['sigma_max_upper']):
            raise ValueError('Stored dual DC bound policy mismatch')
    def solve(value):
        if (not isinstance(value,dict) or value.get('converged') is not True
                or value.get('adjoint') is not False or not isinstance(value.get('rhs'),list)
                or len(value['rhs'])!=1):raise ValueError('Invalid dual DC inverse solve')
        item=value['rhs'][0]
        if not isinstance(item,dict):raise ValueError('Invalid dual DC inverse residual')
        error,threshold=item.get('absolute_residual'),item.get('threshold')
        if (item.get('converged') is not True
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in (error,threshold))
                or not 0<=error<=threshold):raise ValueError('Invalid dual DC inverse residual')
    for index,entry in enumerate(checks):
        if not isinstance(entry,dict) or entry.get('phase')!=('newton' if index<iterations else 'implicit'):
            raise ValueError('Invalid dual DC tangent phase')
        bound(entry.get('bound'),3*cell_count);solve(entry.get('solve'))
    if (current.get('formulation')!='streamed_dual_conservative_cell_current_dc_v1'
            or current.get('operator_derivatives') is not True
            or any(current.get(k) is not False for k in ('global_coupled_matrix_stored','geometry_derivatives',
                'underlying_action_error_certified','physical_qualification','fast_work_qualified'))):
        raise ValueError('Stored dual DC current policy mismatch')
    bound(current.get('coupled_inverse_bound'),6*cell_count);solve(current.get('current_solve'))
    for record,error_key,threshold_key in ((nonlinear,'final_residual_V_per_m_l2','threshold_V_per_m_l2'),
            (nonlinear,'coupled_field_error_V_per_m_l2','threshold_V_per_m_l2'),
            (current,'coupled_current_residual','threshold')):
        error,threshold=record.get(error_key),record.get(threshold_key)
        if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (error,threshold))
                or not 0<=error<=threshold):raise ValueError('Invalid dual DC original residual')
