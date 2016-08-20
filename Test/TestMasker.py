from Util.Masker import Masker

masker = Masker('minc', '/data/data03/quarantine/mni_icbm152_t1_tal_nlin_asym_09a_mask.mnc')
image_masked = masker.mask_image('/data/data03/tharick/perdiff/31_perdiff_unnorm_res.mnc')
masker.save_image_from_data(image_masked, 'Test/MincWrite.mnc')
