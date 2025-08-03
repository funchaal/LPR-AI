from paddleocr import PaddleOCR
import logging

def init_ocr(det_model_dir, rec_model_dir, cls_model_dir=None, use_angle_cls=False, use_openvino=False):
    try:
        ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang='en',
            use_openvino=use_openvino,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir if use_angle_cls else None,
        )
        logging.info("OCR PaddleOCR initialized successfully")
        return ocr
    except Exception as e:
        logging.error(f"Error initializing OCR: {e}")
        raise