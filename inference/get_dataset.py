from dataset import LIDCDataset, XrayLIDCDataset, DEFAULTDataset, XrayCTPADataset, CTPADataset, RSPECTDataset
from torch.utils.data import WeightedRandomSampler
from params import TRAIN_LABELS, VALID_LABELS, RSPECT_TRAIN_LABELS, RSPECT_VALID_LABELS, LIDC_TRAIN_LABELS, LIDC_TEST_LABELS

def get_dataset(cfg):

    if cfg.dataset.name == 'LIDC':
        test_dataset = LIDCDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
        return test_dataset, sampler
    if cfg.dataset.name == 'XRAY_LIDC':
        test_dataset = XrayLIDCDataset(root_dir=cfg.dataset.root_dir, target=LIDC_TEST_LABELS, mode="test")
        sampler = None
        return test_dataset, sampler
    if cfg.dataset.name == 'XRAY_CTPA':
        test_dataset = XrayCTPADataset(root=cfg.dataset.root_dir, target=VALID_LABELS, mode="test")
        sampler = None
        return test_dataset, sampler
    if cfg.dataset.name == 'CTPA':
        test_dataset = CTPADataset(root=cfg.dataset.root_dir, target=VALID_LABELS, mode="test",cond_dim=cfg.model.cond_dim)
        sampler = None
        return test_dataset, sampler
    if cfg.dataset.name == 'RSPECT':
        test_dataset = RSPECTDataset(root_dir=cfg.dataset.root_dir, target=RSPECT_VALID_LABELS, mode="test")
        sampler = None
        return test_dataset, sampler
    if cfg.dataset.name == 'DEFAULT':
        test_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
