import fitz, pytesseract, io, re
from PIL import Image
from src.config import TESSERACT_CMD
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

doc = fitz.open(r"C:\Users\krishkal\Downloads\Document Review Guides\Document Review Guides\3. Supplier Submittals\241135LBP-HRDR001 R0 - Submittal.pdf")

for p_idx in [48, 56]: # Pages 49 and 57
    p = doc[p_idx]
    pix = p.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes()))
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    full_text = " ".join([data["text"][i] for i in range(len(data["text"])) if data["text"][i].strip()])
    
    mat_m = re.search(r"Material\s*:\s*([^\n\r]+?)(?:TYPE|Pipe|Extent|\n|$)", full_text, re.I)
    mat_str = mat_m.group(1).strip() if mat_m else ""
    
    req_m = re.search(r"Required\s*Max\.?\s*Hardness\s*:\s*([^\n\r]+?)(?:Type|Material|\n|$)", full_text, re.I)
    req_str = req_m.group(1).strip() if req_m else ""
    
    is_f22 = bool(re.search(r"(?:F22|P22|WP22|P11|F11|WP11)", mat_str, re.I))
    is_p91 = bool(re.search(r"(?:P91|WP91|F91|Grade\s*91)", mat_str, re.I))
    
    print(f"\nPage {p_idx+1}: mat_str='{mat_str}', is_p91={is_p91}, is_f22={is_f22}, req_str='{req_str}'")
    
    findings = []
    if is_f22 and "280" in req_str:
        box = None
        for i in range(len(data["text"])):
            if data["text"][i].strip() == "280" and data["top"][i] < 300:
                box = (data["left"][i]-3, data["top"][i]-3, data["left"][i]+data["width"][i]+3, data["top"][i]+data["height"][i]+3)
                break
        findings.append({
            "page": p_idx + 1,
            "type": "CRITICAL",
            "comment": f"Page {p_idx+1}: Header specifies Required Max Hardness as '{req_str}' for Alloy 22 component ({mat_str}). Allowable maximum for Alloy 22 (F22) after PWHT is 225 HBW per 700-022 / ASME Section I.",
            "box": box
        })
        
    for i in range(len(data["text"])):
        t = data["text"][i].strip()
        conf = int(data["conf"][i])
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        if y > 500 and conf > 40 and re.match(r"^\d{3}$", t):
            val = int(t)
            if 120 <= val <= 400:
                box = (x - 3, y - 3, x + w + 3, y + h + 3)
                if is_p91 and not is_f22:
                    if val < 195:
                        findings.append({
                            "page": p_idx + 1,
                            "type": "MAJOR",
                            "comment": f"Page {p_idx+1}: Reported hardness reading ({val} HBW) for Alloy 91 is below allowable minimum (195 HBW).",
                            "box": box
                        })
                elif is_f22:
                    if val > 225:
                        findings.append({
                            "page": p_idx + 1,
                            "type": "HOLD",
                            "comment": f"Page {p_idx+1}: Reported hardness reading ({val} HBW) for Alloy 22 exceeds allowable maximum (225 HBW).",
                            "box": box
                        })
                        
    print(f"Total findings on Page {p_idx+1}: {len(findings)}")
    for f in findings:
        print("  ", f["type"], "->", f["comment"])
