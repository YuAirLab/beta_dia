import numpy as np
import torch

from beta_dia import fxic
from beta_dia import param_g
from beta_dia import utils
from beta_dia.log import Logger

try:
    profile
except:
    profile = lambda x: x

logger = Logger.get_logger()

def extract_mall(
        df_batch,
        map_gpu_ms1,
        map_gpu_ms2,
        tol_im,
        tol_ppm,
):
    '''
    Extract top-12 fragment ions mall from ms
    Args:
        df_batch: provide pr info
        map_gpu_ms1: ms
        map_gpu_ms2: ms
        tol_im: tol
        tol_ppm: tol

    Returns:
        Malls: [measure spectrum, bias_im, ppm] * 3, pred, type, area, sa, snr
    '''
    # measure spectrum with smooth
    locus, rts, ims, mzs, xics = fxic.extract_xics(
        df_batch,
        map_gpu_ms1,
        map_gpu_ms2,
        im_tolerance=tol_im,
        ppm_tolerance=tol_ppm,
        cycle_num = 13,
    )

    # [n_pep, n_ion, n_cycle]
    xics = fxic.gpu_simple_smooth(xics)
    ims = utils.convert_numba_to_tensor(ims)[:, 2:, :]
    mzs = utils.convert_numba_to_tensor(mzs)[:, 2:, :]
    xics = utils.convert_numba_to_tensor(xics)[:, 2:, :]

    center_idx = int((xics.shape[-1] - 1) / 2)
    xics_mall = xics[:, :, (center_idx - 1) : (center_idx + 2)]
    xics_mall = xics_mall.permute((0, 2, 1)) # [n_pep, n_cycle, n_ion]
    xics_mall = xics_mall / (torch.amax(xics_mall, dim=-1, keepdim=True) + 1e-7)

    # bias_im
    ims = ims.permute((0, 2, 1))
    ims = ims[:, (center_idx - 1) : (center_idx + 2), :]
    pred_ims = df_batch['pred_im'].values
    pred_ims = torch.from_numpy(pred_ims).to(param_g.gpu_id)
    pred_ims = pred_ims.unsqueeze(-1).unsqueeze(-1).expand(ims.shape)
    bias_ims = pred_ims - ims
    bias_ims[ims < 0] = param_g.tol_im_xic
    bias_ims = bias_ims / param_g.tol_im_xic

    # ppm
    mzs = mzs.permute((0, 2, 1))
    mzs = mzs[:, (center_idx - 1) : (center_idx + 2), :]
    cols_center = ['fg_mz_' + str(i) for i in range(param_g.fg_num)]
    pred_mzs = df_batch[cols_center].values
    pred_mzs = torch.from_numpy(pred_mzs).to(param_g.gpu_id)
    pred_mzs = pred_mzs.unsqueeze(1).expand(mzs.shape)
    ppms = 1e6 * (pred_mzs - mzs) / (pred_mzs + 1e-7)
    ppms[mzs < 1] = param_g.tol_ppm
    ppms = ppms / param_g.tol_ppm

    # sa
    cols = ['score_center_elution_' + str(i) for i in range(14)]
    elutions = df_batch[cols].values[:, 2:]
    elutions = torch.from_numpy(elutions).to(param_g.gpu_id)
    elutions = elutions.unsqueeze(1)

    # area
    locus_start_v = df_batch['score_elute_span_left'].values
    locus_end_v = df_batch['score_elute_span_right'].values
    xics = xics.cpu().numpy()
    mask1 = np.arange(xics.shape[2]) >= locus_start_v[:, None, None]
    mask2 = np.arange(xics.shape[2]) <= locus_end_v[:, None, None]
    xics = xics * mask1 * mask2
    rts = np.repeat(rts[:, np.newaxis, :], xics.shape[1], axis=1)
    areas = np.trapz(xics, x=rts, axis=2)
    areas = areas / (areas.max(axis=1, keepdims=True) + 1e-7)
    areas = torch.from_numpy(areas).to(param_g.gpu_id)
    areas = areas.unsqueeze(1)

    # pred intensities
    cols_height = ['fg_height_' + str(i) for i in range(param_g.fg_num)]
    pred_heights = df_batch[cols_height].values
    pred_heights = torch.from_numpy(pred_heights).to(param_g.gpu_id)
    pred_heights = pred_heights.unsqueeze(1)

    # ion type
    cols_anno = ['fg_anno_' + str(i) for i in range(param_g.fg_num)]
    fg_type = df_batch[cols_anno].values // 1000
    fg_type = torch.from_numpy(fg_type.astype(np.float32)).to(param_g.gpu_id)
    fg_type = fg_type.unsqueeze(1)

    # snr
    cols = ['score_center_snr_' + str(i) for i in range(14)]
    snr = df_batch[cols].values[:, 2:]
    snr = torch.from_numpy(snr).to(param_g.gpu_id)
    snr = snr.unsqueeze(1)

    mall = torch.cat([pred_heights,
                      xics_mall,
                      ppms,
                      bias_ims,
                      fg_type,
                      elutions,
                      areas,
                      snr], dim=1)
    return mall


def scoring_mall(
        model_mall,
        df_input,
        map_gpu_ms1,
        map_gpu_ms2,
        tol_im,
        tol_ppm,
):
    '''
    Extract and score the Malls for elution groups.
    Args:
        model_mall: model
        df_input: provide pr info
        map_gpu_ms1: ms
        map_gpu_ms2: ms
        tol_im: tol
        tol_ppm: tol

    Returns:
        pred, feature
    '''
    mall = extract_mall(df_input,
                        map_gpu_ms1,
                        map_gpu_ms2,
                        tol_im,
                        tol_ppm)
    valid_ion_nums = df_input['fg_num'].values
    valid_ion_nums = torch.from_numpy(valid_ion_nums).long().to(param_g.gpu_id)
    with torch.no_grad():
        feature, pred = model_mall(mall, valid_ion_nums)

    pred = torch.softmax(pred, 1)
    pred = pred[:, 1].cpu().numpy()

    feature = feature.cpu().numpy()

    return pred, feature
