import pytesseract as pt

def detect_plate_string(plate_img=None):
    options='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7'  
    plate_string = pt.image_to_string(plate_img, config=options)[:-1]
    return plate_string