def validate_bounding_box(x1, y1, x2, y2):
    width = x2 - x1
    height = y2 - y1

    if width / height < 16/9:
        return False
    else:
        return True
    
def validate_text(text, min_length=3):
    def is_all_digits(s):
        return s.isdigit()
    
    def is_all_letters(s):
        return s.isalpha()
    
    def is_too_short(s, min_len):
        return len(s) <= min_len

    if is_all_digits(text) or is_all_letters(text) or is_too_short(text, min_length):
        return False
    return True


